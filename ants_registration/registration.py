from __future__ import annotations

import multiprocessing
import os
import pprint
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Union

import ants
import numpy as np
import scipy
from tqdm import tqdm
import yaml
from pyqtgraph.Qt import QtCore

from ants_registration.stack import Stack


class Registration(QtCore.QObject):

    file_path_changed = QtCore.Signal()
    changed = QtCore.Signal()

    result: Dict[str, Any] = None

    def __init__(self):
        QtCore.QObject.__init__(self)

        self.moving = Stack()
        self.fixed = Stack()

        self.file_path: Union[str, None] = None

        self.settings: Dict[str, Any] = default_reg_settings.copy()

    @property
    def save_path(self):
        """Return the path where registration results and metadata are saved
           (relative to the directory containing the moving stack)
        """
        # Create save path based on moving and fixed file paths
        moving_name = self.moving.file_path.split('/')[-1]
        reference_name = self.fixed.file_path.split('/')[-1]

        # Combine and create
        path = '/'.join(['ants_registration', moving_name, reference_name])

        if not os.path.exists(path):
            print(f'Create save path {path}')
            os.makedirs(path, exist_ok=True)

        return path

    def data_complete(self) -> bool:
        """Return true if both reference and moving stack are set"""
        return self.fixed.file_path is not None and self.moving.file_path is not None

    def exists(self) -> bool:
        """Return true of there is a Composite.h5 file (created by ANTs) on the save path"""
        return os.path.exists(f'{self.save_path}/Composite.h5')

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

        # Get rotation transform
        tr_rot_x_ants = self.get_ants_x_rotation_transform()

        # Get rotation transform
        tr_rot_y_ants = self.get_ants_y_rotation_transform()

        # Get rotation transform
        tr_rot_z_ants = self.get_ants_z_rotation_transform()

        # Create translation transform
        x_offset = (fixed_scale[0] * fixed_shape[0] - moving_scale[0] * moving_shape[0])
        T = np.array([[1, 0, 0, _trans[0] - x_offset],
                      [0, 1, 0, -_trans[1]],
                      [0, 0, 1, -_trans[2]]])

        tr_trans_ants = ants.create_ants_transform(transform_type='AffineTransform')
        tr_trans_ants.set_parameters(T)

        # Combine transforms
        tr_ants = ants.compose_ants_transforms([tr_trans_ants, tr_rot_x_ants, tr_rot_y_ants, tr_rot_z_ants])

        return [tr_ants, tr_trans_ants, tr_rot_x_ants, tr_rot_y_ants, tr_rot_z_ants]

    def get_ants_x_rotation_transform(self):

        _rot = np.deg2rad(self.moving.x_rotation)

        moving_scale = self.moving.resolution
        moving_shape = self.moving.shape

        # Create rotation
        c_rot = np.array([moving_shape[0] / 2, moving_shape[1] / 2, moving_shape[2] / 2])
        R = np.array([[1, 0, 0, 0],
                      [0, np.cos(_rot), -np.sin(_rot), 0],
                      [0, np.sin(_rot), np.cos(_rot), 0]])

        tr_rot_ants = ants.create_ants_transform(transform_type='AffineTransform')
        tr_rot_ants.set_parameters(R)
        tr_rot_ants.set_fixed_parameters(c_rot * moving_scale)

        return tr_rot_ants

    def get_ants_y_rotation_transform(self):

        _rot = np.deg2rad(self.moving.y_rotation)

        moving_scale = self.moving.resolution
        moving_shape = self.moving.shape

        # Create rotation
        c_rot = np.array([moving_shape[0] / 2, moving_shape[1] / 2, moving_shape[2] / 2])
        R = np.array([[np.cos(_rot), 0, np.sin(_rot), 0],
                      [0, 1, 0, 0],
                      [-np.sin(_rot), 0, np.cos(_rot), 0]])

        tr_rot_ants = ants.create_ants_transform(transform_type='AffineTransform')
        tr_rot_ants.set_parameters(R)
        tr_rot_ants.set_fixed_parameters(c_rot * moving_scale)

        return tr_rot_ants

    def get_ants_z_rotation_transform(self):

        _rot = -np.deg2rad(self.moving.z_rotation)

        moving_scale = self.moving.resolution
        moving_shape = self.moving.shape

        # Create rotation
        c_rot = np.array([moving_shape[0] / 2, moving_shape[1] / 2, moving_shape[2] / 2])
        R = np.array([[np.cos(_rot), -np.sin(_rot), 0, 0],
                      [np.sin(_rot), np.cos(_rot), 0, 0],
                      [0, 0, 1, 0]])

        tr_rot_ants = ants.create_ants_transform(transform_type='AffineTransform')
        tr_rot_ants.set_parameters(R)
        tr_rot_ants.set_fixed_parameters(c_rot * moving_scale)

        return tr_rot_ants

    def get_alignment_rgb_stack(self) -> np.ndarray:

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
        raw_aligned_image = tr_ants.apply_to_image(moving_stack_ants, fixed_stack_ants, interpolation='nearestneighbor')

        # Create image
        image_data = np.concatenate(
            (fixed_stack_ants.numpy().astype(self.fixed.data.dtype)[:, :, :, np.newaxis],
             raw_aligned_image.numpy().astype(self.moving.data.dtype)[:, :, :, np.newaxis],
             np.zeros(fixed_stack_ants.shape, dtype=self.fixed.data.dtype)[:, :, :, np.newaxis]),
            axis=3
        )

        return image_data

    def get_registered_rgb_stack(self) -> np.ndarray:

        # Apply transform
        fixed_stack_ants = ants.from_numpy(self.fixed.data, spacing=(*self.fixed.resolution,))
        moving_stack_ants = ants.from_numpy(self.moving.data, spacing=(*self.moving.resolution,))
        warped_stack_ants = ants.apply_transforms(fixed_stack_ants, moving_stack_ants,
                                                  [f'{self.save_path}/Composite.h5'],
                                                  interpolator='nearestNeighbor')

        fixed_stack = fixed_stack_ants.numpy().astype(self.fixed.data.dtype)
        warped_stack = warped_stack_ants.numpy().astype(self.moving.data.dtype)

        image_data = np.concatenate(
            (fixed_stack[:, :, :, np.newaxis],
             warped_stack[:, :, :, np.newaxis],
             np.zeros(fixed_stack.shape, dtype=self.fixed.data.dtype)[:, :, :, np.newaxis]),
            axis=3
        )

        return image_data

    def run_pre_alignment(self, moving_image: np.ndarray):
        """Estimate rough z-alignment of suite2p's mean image to the fixed stack, using phase correlation
        """

        print('Run pre alignment for reference layers')

        def phase_correlations(ref: np.ndarray, im: np.ndarray) -> np.ndarray:
            """Phase correlation calculation
            after: https://github.com/michaelting/Phase_Correlation/blob/master/phase_corr.py

            Parameters
            ----------
            ref : array
                Array of shape (N, M)
            im : array
                Array of shape (N, M)

            Returns
            -------
            array
                Array of same size as ref, containing the phase correlations for the
                corresponding index shift.
            """

            conj = np.ma.conjugate(np.fft.fft2(im))
            r = np.fft.fft2(ref) * conj
            r /= np.absolute(r)
            return np.fft.ifft2(r).real

        # Get reference stack
        zstack = self.fixed.data

        # Get s2p mean image and resample to target dimensions
        # moving_im_ants = ants.from_numpy(self.moving.s2p_mean_image[:, :, None], spacing=(*self.moving.resolution,))
        moving_im_ants = ants.from_numpy(moving_image[:, :, None], spacing=(*self.moving.resolution,))
        zstack_ants = ants.from_numpy(zstack, spacing=(*self.fixed.resolution,))
        # Get and apply rotation
        # _rot_transform = self.get_ants_rotation_transform()
        # moving_im_ants = _rot_transform.apply_to_image(moving_im_ants, zstack_ants)
        # moving_im = moving_im_ants.numpy()[:, :, 0]
        moving_im = ants.resample_image_to_target(moving_im_ants, zstack_ants).numpy()[:, :, 0]

        # Determine padding and make sure it is divisible by 2
        padding = moving_im.shape[0] / 4
        padding = int(padding // 2 * 2)

        # Pad reference on all sides
        moving_im_pad = np.pad(moving_im, (padding // 2, padding // 2))

        corrs = []
        xy = []
        for i in tqdm(range(zstack.shape[2])):
            ref_image = np.pad(zstack[:, :, i], (0, padding))

            corrimg = phase_correlations(ref_image, moving_im_pad)

            # Smoothen phase correlation image to make maximum-estimate more robust
            corrimg_sm = scipy.ndimage.gaussian_filter(corrimg, sigma=3)

            # Get maximum phase correlation and respective x/y shifts
            maxcorr = corrimg_sm.max()
            x, y = np.unravel_index(corrimg_sm.argmax(), corrimg_sm.shape)

            x -= padding // 2
            y -= padding // 2

            corrs.append(maxcorr)
            xy.append([x, y])

        return corrs, xy

    def run(self):

        settings = self.settings.copy()

        print('Run registration')

        # Get Image data
        fixed_stack_ants = ants.from_numpy(self.fixed.data, spacing=(*self.fixed.resolution,))
        moving_stack_ants = ants.from_numpy(self.moving.data, spacing=(*self.moving.resolution,))

        # Get transforms
        transforms = self.get_ants_init_transform()

        if transforms is None:
            init_transforms = None
        else:
            trans_path = f'{self.save_path}/init_trans.mat'
            ants.write_transform(transforms[1], trans_path)
            rot_x_path = f'{self.save_path}/init_rot_x.mat'
            ants.write_transform(transforms[2], rot_x_path)
            rot_y_path = f'{self.save_path}/init_rot_y.mat'
            ants.write_transform(transforms[3], rot_y_path)
            rot_z_path = f'{self.save_path}/init_rot_z.mat'
            ants.write_transform(transforms[4], rot_z_path)

            init_transforms = [trans_path, rot_x_path, rot_y_path, rot_z_path]

        # Update settings
        settings['initial_transform'] = init_transforms
        settings['verbose'] = True
        settings['write_composite_transform'] = True
        settings['outprefix'] = f'{self.save_path}/'
        # Make sure that these are tuples
        #  ANTs does not like lists for these and yaml doesn't do tuples by default
        settings['reg_iterations'] = tuple(settings['reg_iterations'])
        settings['aff_iterations'] = tuple(settings['aff_iterations'])
        settings['aff_shrink_factors'] = tuple(settings['aff_shrink_factors'])
        settings['aff_smoothing_sigmas'] = tuple(settings['aff_smoothing_sigmas'])

        meta = {'fixed_resolution': [float(f) for f in self.fixed.resolution],
                'fixed_path': Path(os.path.relpath(self.fixed.file_path)).as_posix(),
                'moving_resolution': [float(f) for f in self.moving.resolution],
                'init_translation': [float(f) for f in self.moving.translation],
                'init_x_rotation': float(self.moving.x_rotation),
                'init_y_rotation': float(self.moving.y_rotation),
                'init_z_rotation': float(self.moving.z_rotation),
                'registration_settings': settings}

        yaml.safe_dump(meta, open(f'{self.save_path}/metadata.yaml', 'w'))

        # result = ants.registration(fixed_stack_ants, moving_stack_ants, **settings)

        # Run registration
        proc = multiprocessing.Process(target=run_ants_registration,
                                       name=f'ANTS registration run',
                                       args=(self.save_path, fixed_stack_ants, moving_stack_ants),
                                       kwargs=settings)
        proc.start()
        proc.join()

        self.changed.emit()


def run_ants_registration(save_path: str, *args, **kwargs):
    # result = ants.registration(*args, **kwargs)
    # TODO: using ANTsLog causes the call to ants.registration to fail when using CMD call "antsui"
    # with ANTsLog(f'{save_path}/registration.log') as log:

    with open(f'{save_path}/registration.log', 'w') as sys.stdout:
        # Run ANTS registration
        result = ants.registration(*args, **kwargs)
        pprint.pprint(result)



# class ANTsLog(object):
#     """
#     Re-directs the ANTs verbose output to a textfile
#     Method derived from Maximilian Hoffman's solution:
#     https://github.com/ANTsX/ANTsPy/issues/130
#     """
#
#     def __init__(self, log_fpath):
#         self.log_fpath = log_fpath
#         self.redirect()
#
#     def redirect(self):
#         # get file descriptor to __stdout__ (__stdout__ used instead of stdout,
#         # because JupyterLab modifies stdout)
#         self.orig_stdout_fd = sys.__stdout__.fileno()
#         # Duplicate file descriptor to __stdout__
#         self.saved_stdout_fd = os.dup(self.orig_stdout_fd)
#         # create logfile and redirect __stdout__
#         # Log should be unique for each unique registration,
#         # so file is always newly-created
#         self.log = open(self.log_fpath, "wb")
#         os.dup2(self.log.fileno(), self.orig_stdout_fd)
#
#     def revert(self):
#         """
#         Put everything back as it was
#         """
#         # Close log
#         self.log.close()
#         # reset __stdout__
#         os.dup2(self.saved_stdout_fd, self.orig_stdout_fd)
#
#     def __enter__(self):
#         # enter method added so clas can be used
#         # with context manager
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # exit method added so clas can be used
#         # with context manager
#         self.revert()


# class ANTsLog(object):
#     """
#     Simplified class after PichardRarker's solution, that
#     Re-directs the ANTs verbose output to a textfile
#     Method derived from Maximilian Hoffman's solution:
#     https://github.com/ANTsX/ANTsPy/issues/130
#     """
#
#     def __init__(self, log_fpath):
#         self.log_fpath = log_fpath
#
#         # get file descriptor to __stdout__ (__stdout__ used instead of stdout)
#         self.original_stdout = sys.__stdout__.fileno()
#
#         # Duplicate file descriptor to __stdout__
#         self.saved_stdout = os.dup(self.original_stdout)
#
#         self.log_file = open(self.log_fpath, "wb")
#         os.dup2(self.log_file.fileno(), self.original_stdout)
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.log_file.close()
#         # reset __stdout__
#         os.dup2(self.saved_stdout, self.original_stdout)


# Default settings
default_reg_settings = dict(
        type_of_transform="Affine",
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

# Instantiate
registration = Registration()

"""
Copyright 2018 ICG, Graz University of Technology

This file is part of MURAUER.

MURAUER is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MURAUER is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MURAUER.  If not, see <http://www.gnu.org/licenses/>.
"""

# Project specific
import data.basetypes as basetypes
from data.loaderfactory import DatasetType
from data.NyuHandPoseDataset import NyuAnnoType

# External libs
import numpy as np


args_data = basetypes.Arguments()

args_data.dataset_type = DatasetType.NYU

# Change to point to the original NYU dataset
args_data.nyu_data_basepath = "/path/to/NyuDataset/original_data"

# If a "cache" should be used (=> faster loading/training), change the path
args_data.use_pickled_cache = False
args_data.nyu_data_basepath_pickled = "/path/to/NyuDataset/original_data_pickled"

# frame IDs (0-based)
args_data.id_start_train, args_data.id_end_train = 0, 72756
args_data.setup_swap_id_train = 29116       # first ID after camera views are swaped, i.e., view 2 and 3 swaped
args_data.id_start_val, args_data.id_end_val = 0, 2439
# Number of samples in full train set (i.e., indexable)
args_data.num_all_samples_train = 72757     # this applies to the frame-IDs, i.e., if a sample (with the same ID) is used from several camera-views it is only counted once
# Number of used samples from validation set
args_data.max_val_train_ratio = 0.3   # ratio of validation samples over train samples
args_data.max_num_samples_val = 2440  # maximum number of used validation samples

args_data.cam_ids_for_pose_train_real = np.asarray([1], dtype=np.int32)
args_data.cam_ids_for_pose_train_synth = np.asarray([1,2,3], dtype=np.int32)
args_data.output_cam_ids_train = np.asarray([3])    # for view prediction
args_data.output_cam_ids_test = np.asarray([2])     # for view prediction
# needed_cam_ids should include all cameras to be loaded, no matter for what they are used (e.g., pose estimation training/pose loss, view prediction, ...)
args_data.needed_cam_ids_train_real = args_data.cam_ids_for_pose_train_real
args_data.needed_cam_ids_train_synth = args_data.cam_ids_for_pose_train_synth
args_data.needed_cam_ids_test = [1]

# Which joints to use (EVAL_JOINTS_ORIGINAL, ALL_JOINTS)
args_data.anno_type = NyuAnnoType.EVAL_JOINTS_ORIGINAL

args_data.num_joints = args_data.anno_type

args_data.in_crop_size = (128, 128)

# Jitter parameters; per cam ([cam1, cam2, cam3]); 
# assumed to be the same for view 2 and 3 when swaping views, e.g., for preview
# Center of mass (com)
args_data.do_jitter_com = [True, True, True]
args_data.do_jitter_com_test = [False, False, False]
args_data.sigma_com = np.asarray([5., 5., 5.])  # in millimeter; for normal distribution
# White noise
args_data.do_add_white_noise = [True, True, True]
args_data.do_add_white_noise_test = [False, False, False]
args_data.sigma_noise = [5., 5., 5.]            # in millimeter; for normal distribution
# Cube-size
args_data.do_jitter_cubesize = [False, False, False]
args_data.do_jitter_cubesize_test = [False, False, False]
args_data.sigma_cubesize = [15., 15., 15.]      # in millimeter; for uniform distribution, range [-sigma, +sigma]
# Rotation
args_data.do_jitter_rotation = [True, True, True]
args_data.do_jitter_rotation_test = [False, False, False]
args_data.rotation_angle_range = [-60.0, 60.0]  # for uniform distribution, range: [min, max]

# Minimum ratio of the detected hand crop which should be inside image boundaries
args_data.min_ratio_inside = 0.3

# value normalization: \in [0,1] if True, \in [-1,1] otherwise
args_data.do_norm_zero_one = False

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

from __future__ import print_function
import cv2 # to be imported before torch (for me; so that libgomp is loaded from the system installation)
# Project specific
from util.argparse_helper import parse_arguments_generic
import data.loaderfactory as loaderfactory
from data.basetypes import LoaderMode
# Other libs
import numpy as np
import progressbar as pb


#%% Set configuration
do_generate_train_set = True
do_generate_test_set = True

from config.config import args                  # General parameters
args = parse_arguments_generic(args)            # Parse command-line arguments
from config.config_data_nyu import args_data    # Dataset specific parameters
# Merge different configuration parameters into single object
args.__dict__ = dict(args.__dict__.items() + args_data.__dict__.items())

# Which cams to load
args.cam_ids_for_pose_train_real = np.asarray([1,2,3], dtype=np.int32)
args.cam_ids_for_pose_train_synth = np.asarray([1,2,3], dtype=np.int32)
args.needed_cam_ids_train_real = args.cam_ids_for_pose_train_real
args.needed_cam_ids_train_synth = args.cam_ids_for_pose_train_synth
args.needed_cam_ids_test = [1,2,3]

args.use_pickled_cache = True
args.batch_size = 128
args.num_loader_workers = 5
    

#%% Load all samples
if do_generate_train_set:
    print("Load train set and generate binary files if missing")
    train_loader = loaderfactory.create_sequential_nyu_trainloader(args)
    bar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar()], maxval=len(train_loader))
    bar.start()
    print("Loading samples...")
    for i, (img_c1_r, img_c2_r, img_c3_r, target_c1_r, target_c2_r, target_c3_r, \
            transform_crop_c1_r, transform_crop_c2_r, transform_crop_c3_r, \
            com_c1_r, com_c2_r, com_c3_r, \
            size_c1_r, size_c2_r, size_c3_r, \
            img_c1_s, img_c2_s, img_c3_s, target_c1_s, target_c2_s, target_c3_s, \
            transform_crop_c1_s, transform_crop_c2_s, transform_crop_c3_s, \
            com_c1_s, com_c2_s, com_c3_s, \
            size_c1_s, size_c2_s, size_c3_s, is_l) \
            in enumerate(train_loader):
        # Do something
        l1r, l2r, l3r = len(img_c1_r), len(img_c2_r), len(img_c3_r)
        l1s, l2s, l3s = len(img_c1_s), len(img_c2_s), len(img_c3_s)
        assert l1r == l2r == l3r == l1s == l2s == l3s > 0, "batches not equal or empty"
        bar.update(i+1)
        
    bar.finish()
            
if do_generate_test_set:
    print("Load test set and generate binary files if missing")
    test_loader = loaderfactory.create_dataloader(LoaderMode.TEST, args)
    bar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar()], maxval=len(test_loader))
    bar.start()
    print("Loading samples...")
    for i, (img_c1_r, img_c2_r, img_c3_r, target_c1_r, target_c2_r, target_c3_r, \
            transform_crop_c1_r, transform_crop_c2_r, transform_crop_c3_r, \
            com_c1_r, com_c2_r, com_c3_r, \
            size_c1_r, size_c2_r, size_c3_r, \
            img_c1_s, img_c2_s, img_c3_s, target_c1_s, target_c2_s, target_c3_s, \
            transform_crop_c1_s, transform_crop_c2_s, transform_crop_c3_s, \
            com_c1_s, com_c2_s, com_c3_s, \
            size_c1_s, size_c2_s, size_c3_s, is_l) \
            in enumerate(test_loader):
        # Do something
        l1r, l2r, l3r = len(img_c1_r), len(img_c2_r), len(img_c3_r)
        assert l1r == l2r == l3r > 0, "batches not equal or empty"
        bar.update(i+1)
        
    bar.finish()

if not do_generate_train_set and not do_generate_test_set:
    print("Nothing to be done (both generation flags set false).")
    
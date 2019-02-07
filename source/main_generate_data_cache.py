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

from util.argparse_helper import parse_arguments_generic
from data.NyuHandPoseDataset import NyuHandPoseMultiViewDataset#, NyuHandPoseDataset #, NyuAnnoType
#from data.IcgHandPoseDataset import IcgHandPoseMultiViewDataset
from data.basetypes import LoaderMode, DatasetType

# PyTorch
import torch
from torchvision import transforms
import torch.utils.data

import numpy as np
import progressbar


#%% Set configuration
do_generate_train_set = True
do_generate_test_set = True

# General parameters
from config.config import args
# Parse command-line arguments
args = parse_arguments_generic(args)
# Dataset specific parameters
if args.dataset_type == DatasetType.NYU:
    from config.config_data_nyu import args_data
elif args.dataset_type == DatasetType.ICG:
    from config.config_data_icg import args_data    
# Merge different configuration parameters into single object
args.__dict__ = dict(args.__dict__.items() + args_data.__dict__.items())

# Which cams to load
args.cam_ids_for_pose_train_real = np.asarray([1,2,3], dtype=np.int32)
args.cam_ids_for_pose_train_synth = np.asarray([1,2,3], dtype=np.int32)
args.needed_cam_ids_train_real = args.cam_ids_for_pose_train_real
args.needed_cam_ids_train_synth = args.cam_ids_for_pose_train_synth
args.needed_cam_ids_test = [1,2,3]

args.batch_size = 128
args.num_loader_workers = 3


#%% Helper
def create_loader(args_data, loader_type):
    kwargs = {'num_workers': args_data.num_loader_workers}
    
    if loader_type == LoaderMode.TRAIN:
        loader = torch.utils.data.DataLoader(
            NyuHandPoseMultiViewDataset(args_data.nyu_data_basepath, train=True, 
                                        cropSize=args_data.in_crop_size,
                                        doJitterCom=args_data.do_jitter_com,
                                        sigmaCom=args_data.sigma_com,
                                        doAddWhiteNoise=args_data.do_add_white_noise,
                                        sigmaNoise=args_data.sigma_noise,
                                        transform=transforms.ToTensor(),
                                        useCache=args_data.use_pickled_cache,
                                        cacheDir=args_data.nyu_data_basepath_pickled, 
                                        annoType=args_data.anno_type,
                                        neededCamIdsReal=args_data.needed_cam_ids_train_real,
                                        neededCamIdsSynth=args_data.needed_cam_ids_train_synth,
                                        randomSeed=args_data.seed,
                                        cropSize3D=args_data.crop_size_3d_tuple,
                                        args_data=args_data),
            batch_size=args_data.batch_size,
            **kwargs)
                
    elif loader_type == LoaderMode.TEST:
        needed_cam_ids_synth = args_data.needed_cam_ids_test if args_data.do_test_on_synth else []
        
        loader = torch.utils.data.DataLoader(
            NyuHandPoseMultiViewDataset(args_data.nyu_data_basepath, train=False, 
                                        cropSize=args_data.in_crop_size,
                                        doJitterCom=args_data.do_jitter_com_test,
                                        sigmaCom=args_data.sigma_com,
                                        doAddWhiteNoise=args_data.do_add_white_noise_test,
                                        sigmaNoise=args_data.sigma_noise,
                                        transform=transforms.ToTensor(),
                                        useCache=args_data.use_pickled_cache,
                                        cacheDir=args_data.nyu_data_basepath_pickled, 
                                        annoType=args_data.anno_type,
                                        neededCamIdsReal=args_data.needed_cam_ids_test,
                                        neededCamIdsSynth=needed_cam_ids_synth,
                                        randomSeed=args_data.seed,
                                        cropSize3D=args_data.crop_size_3d_tuple,
                                        args_data=args_data),
            batch_size=args_data.batch_size,
            **kwargs)
                    
        print("Using {} samples for test".format(len(loader.sampler)))
        
    else:
        raise UserWarning("LoaderMode unknown.")
            
    return loader
    

#%% Load all samples
if do_generate_train_set:
    print("Loading train set...")
    train_loader = create_loader(args, loader_type=LoaderMode.TRAIN)
    widgets = [progressbar.Percentage(), progressbar.Bar()]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=len(train_loader))
    bar.start()
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
            
if do_generate_test_set:    
    print("Loading test set...")
    test_loader = create_loader(args, loader_type=LoaderMode.TEST)
    widgets = [progressbar.Percentage(), progressbar.Bar()]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=len(test_loader))
    bar.start()
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
        l1s, l2s, l3s = len(img_c1_s), len(img_c2_s), len(img_c3_s)
        assert l1r == l2r == l3r == l1s == l2s == l3s > 0, "batches not equal or empty"
        bar.update(i+1)

if not do_generate_train_set and not do_generate_test_set:
    print("Nothing to be done (both generation flags set false).")
    
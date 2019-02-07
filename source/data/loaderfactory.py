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
from data.NyuHandPoseDataset import NyuHandPoseMultiViewDataset, NyuHandPoseDataset
from data.basetypes import LoaderMode, DatasetType, TrainingType, NetType, TrainLoaders

# PyTorch
import torch
from torchvision import transforms
import torch.utils.data.sampler as smpl

# Libs
import numpy as np
import copy


#%% Functions
def create_dataloader(loader_type, args):
    """
    Create a data loader according to the parameters
    """
    
    kwargs = {'num_workers': args.num_loader_workers, 'pin_memory': True} if args.cuda else {}
    
    if args.dataset_type == DatasetType.NYU:
        if loader_type == LoaderMode.VAL:
            print("===> Creating validation data loader...")
            needed_cam_ids = np.append(args.output_cam_ids_test, args.needed_cam_ids_test)
            needed_cam_ids_synth = []
            
            num_samples_val = min(int(round(args.max_val_train_ratio * args.num_labeled_samples)), 
                                  args.max_num_samples_val)
            ids_val = np.arange(args.id_start_val, args.id_end_val+1)
            ids_val = args.rng.permutation(ids_val)
            ids_val = ids_val[:num_samples_val]
            loader = torch.utils.data.DataLoader(
                NyuHandPoseMultiViewDataset(args.nyu_data_basepath, train=False, 
                                            cropSize=args.in_crop_size,
                                            doJitterCom=args.do_jitter_com_test,
                                            sigmaCom=args.sigma_com,
                                            doAddWhiteNoise=args.do_add_white_noise_test,
                                            sigmaNoise=args.sigma_noise,
                                            transform=transforms.ToTensor(),
                                            useCache=args.use_pickled_cache,
                                            cacheDir=args.nyu_data_basepath_pickled, 
                                            annoType=args.anno_type,
                                            neededCamIdsReal=needed_cam_ids,
                                            neededCamIdsSynth=needed_cam_ids_synth,
                                            randomSeed=args.seed,
                                            cropSize3D=args.crop_size_3d_tuple,
                                            args_data=args),
                batch_size=args.batch_size,
                sampler=smpl.SubsetRandomSampler(ids_val), **kwargs)
                        
            print("Using {} samples for validation".format(len(ids_val)))
                
        elif loader_type == LoaderMode.TEST:
            print("===> Creating test data loader...")
            needed_cam_ids_synth = []
            
            loader = torch.utils.data.DataLoader(
                NyuHandPoseMultiViewDataset(args.nyu_data_basepath, train=False, 
                                            cropSize=args.in_crop_size,
                                            doJitterCom=args.do_jitter_com_test,
                                            sigmaCom=args.sigma_com,
                                            doAddWhiteNoise=args.do_add_white_noise_test,
                                            sigmaNoise=args.sigma_noise,
                                            transform=transforms.ToTensor(),
                                            useCache=args.use_pickled_cache,
                                            cacheDir=args.nyu_data_basepath_pickled, 
                                            annoType=args.anno_type,
                                            neededCamIdsReal=args.needed_cam_ids_test,
                                            neededCamIdsSynth=needed_cam_ids_synth,
                                            randomSeed=args.seed,
                                            cropSize3D=args.crop_size_3d_tuple,
                                            args_data=args),
                batch_size=args.batch_size,
                **kwargs)
                        
            print("Using {} samples for test".format(len(loader.sampler)))
            
        else:
            raise UserWarning("LoaderMode implemented.")
            
    else:
        raise UserWarning("DatasetType not implemented.")
            
    return loader
    
    
def create_train_loaders(args):
    """
    Create all train loaders
    """
    # Sanity check
    if not (args.net_type == NetType.HAPE_RESNET50_MAP_PREVIEW) \
            or not (args.training_type == TrainingType.ADVERSARIAL):
        raise UserWarning("Training/Net-Type combination not implemented.")
            
    print("===> Creating train data loaders...")
    train_loaders = create_train_dataloaders_map_preview_adv(LoaderMode.TRAIN, args)
                    
    return train_loaders
    
    
def create_train_dataloaders_map_preview_adv(loader_type, args_data):
    """
    Create the specific data loaders used for training
    """
    # Sanity check
    if not loader_type == LoaderMode.TRAIN:
        print("Required loader-type {} not implemented.".format(loader_type))
        raise UserWarning("Requested loaders only implemented for TRAINING.")
        
    # Max. #workers per train sample (to limit interference if few samples)
    max_ratio_workers = 0.02
    args_data.min_samp_prob = 1.0   # here this is only used for labeled data
    
    do_use_gpu = args_data.cuda
        
    # Normalize ratios
    sum_ratios = float(args_data.ratio_corresponding_data_in_batch \
                        + args_data.ratio_synth_data_in_batch \
                        + args_data.ratio_unlabeled_data_in_batch)
    ratio_corr = args_data.ratio_corresponding_data_in_batch / sum_ratios
    ratio_synt = args_data.ratio_synth_data_in_batch / sum_ratios
    
    print("Creating loader for SYNTHETIC PRE-TRAINING data")
    loader_pretrain = create_loader_synth_data(loader_type, args_data, do_use_gpu, 
                                               args_data.num_loader_workers,
                                               batch_size=args_data.batch_size, 
                                               seed=args_data.seed)
    
    print("Creating loader for corresponding REAL<->SYNTHETIC data")
    num_corr = int(np.round(args_data.batch_size * ratio_corr))
    # Ensure reasonable number of workers
    num_workers_corr = min(int(np.round(args_data.num_loader_workers * ratio_corr)), 
                           int(np.round(args_data.num_labeled_samples * max_ratio_workers)))
#    num_workers_corr = max(1, num_workers_corr)
    loader_corr, ids_train_permuted = create_loader_corresponding_real_synth_data(
                                                                loader_type, 
                                                                args_data, 
                                                                do_use_gpu, 
                                                                num_workers_corr,
                                                                batch_size=num_corr)
    
    print("Creating loader for separate SYNTHETIC data")
    num_synth = int(np.round(args_data.batch_size * ratio_synt))
    num_workers_synth = int(np.round(args_data.num_loader_workers * ratio_synt))
#    num_workers_synth = max(1, num_workers_synth)
    loader_synth = create_loader_synth_data(loader_type, args_data, do_use_gpu, 
                                            num_workers_synth,
                                            batch_size=num_synth, 
                                            seed=args_data.seed-2)
    
    print("Creating loader for view prediction, unlabeled REAL and SYNTHETIC data")
    num_ul_prev = int(np.round((args_data.batch_size - num_corr - num_synth) / 2.0))
    num_ul_prev = max(1, num_ul_prev)
    num_workers_ul_prev = int(np.round((args_data.num_loader_workers - num_workers_corr \
                                        - num_workers_synth) / 2.0))
#    num_workers_ul_prev = max(1, num_workers_ul_prev)
    cam_ids_corr = np.intersect1d(args_data.cam_ids_for_pose_train_real, 
                                  args_data.cam_ids_for_pose_train_synth)
    if len(cam_ids_corr) > 1:
        raise UserWarning("Only one corresponding cam assumed during training \
            (corr. view point between real and synth. data, same view, \
            similar distribution for adv. training and view prediction)")
    # No rotation jitter, no cubesize jitter
    used_device = args_data.used_device
    args_data.used_device = None    # there were issues with copying the device; ignoring it - it's not needed here
    args_data_temp = copy.deepcopy(args_data)
    args_data_temp.do_jitter_rotation = [False, False, False]
    args_data_temp.do_jitter_cubesize = [False, False, False]
    args_data_temp.sigma_com[args_data.output_cam_ids_train - 1] = 0.0
    min_samp_prob_labeled = 0.3
    needed_cam_ids = np.append(args_data_temp.output_cam_ids_train, 1)    # input cam ID is always 1 for now
    id_range = [args_data.id_start_train, args_data.id_end_train+1]
    loader_preview = create_loader_preview(loader_type, args_data_temp, do_use_gpu, 
                                      num_workers_ul_prev, min_samp_prob_labeled, num_ul_prev,
                                      needed_cam_ids, needed_cam_ids, args_data.seed-3, 
                                      id_range,
                                      ids_train_permuted=ids_train_permuted)
    
    args_data.used_device = used_device     # restoring device
                                      
    print("Creating loaders for NON-corresponding, unlabeled REAL and SYNTHETIC data")
    num_ul_wc = args_data.batch_size - num_corr - num_synth - num_ul_prev
    num_ul_wc = max(1, num_ul_wc)
    num_workers_ul1 = int(np.round((args_data.num_loader_workers - num_workers_corr \
                                    - num_workers_synth - num_workers_ul_prev) / 2.0))
#    num_workers_ul1 = max(1, num_workers_ul1)
    num_workers_ul2 = args_data.num_loader_workers - num_workers_corr \
                        - num_workers_synth - num_workers_ul_prev - num_workers_ul1
#    num_workers_ul2 = max(1, num_workers_ul2)
    num_labeled_samples_ul = 0
    min_samp_prob_ul = 0.0
    print("  REAL")
    loader_real_weakcorr_ul = create_independent_data_loader(loader_type, 
                                                args_data, 
                                                do_use_gpu,
                                                num_workers_ul1,
                                                num_labeled_samples_ul, 
                                                min_samp_prob=min_samp_prob_ul, 
                                                batch_size=num_ul_wc, 
                                                cam_ids_real=cam_ids_corr, 
                                                cam_ids_synth=[], 
                                                seed=args_data.seed-4)
    print("  SYNTH")
    loader_synth_weakcorr_ul = create_independent_data_loader(loader_type, 
                                                args_data, 
                                                do_use_gpu, 
                                                num_workers_ul2,
                                                num_labeled_samples_ul, 
                                                min_samp_prob=min_samp_prob_ul, 
                                                batch_size=num_ul_wc, 
                                                cam_ids_real=[], 
                                                cam_ids_synth=cam_ids_corr, 
                                                seed=args_data.seed-5)
            
    return TrainLoaders(train_loader=[],
                        loader_pretrain=loader_pretrain, 
                        loader_corr=loader_corr, 
                        loader_real=[],
                        loader_synth=loader_synth, 
                        loader_real_weakcorr_ul=loader_real_weakcorr_ul,
                        loader_synth_weakcorr_ul=loader_synth_weakcorr_ul,
                        loader_preview=loader_preview)
            
            
def create_loader_corresponding_real_synth_data(loader_type, args_data, do_use_gpu, 
                                                num_loader_workers, batch_size,
                                                id_range=[]):
    # Sanity check
    if not loader_type == LoaderMode.TRAIN:
        print("Required loader-type {} not implemented.".format(loader_type))
        raise UserWarning("requested loader generation currently only implemented for TRAINING data loaders")
        
    kwargs = {'num_workers': num_loader_workers, 'pin_memory': True} if do_use_gpu else {}
        
    if args_data.dataset_type == DatasetType.NYU:
        cam_ids_r = args_data.cam_ids_for_pose_train_real
        cam_ids_s = args_data.cam_ids_for_pose_train_synth
        cam_ids_corr = np.intersect1d(cam_ids_r, cam_ids_s)
        
        if len(id_range) == 0:
            id_range = [args_data.id_start_train, args_data.id_end_train+1]
        
        # Set up sample IDs to sample from
        ids_train = np.arange(*id_range)
        ids_train_permuted = args_data.rng.permutation(ids_train)
        ids_train_labeled = ids_train_permuted[:args_data.num_labeled_samples]
        ids_train_unlabeled = ids_train_permuted[args_data.num_labeled_samples:]
        # Ensure a minimum sampling probability for labeled samples
        ratio_labeled = len(ids_train_labeled) / float(len(ids_train))
        prob_labeled = max(args_data.min_samp_prob, ratio_labeled)
        prob_unlabeled = 1.0 - prob_labeled
        # Set up distribution/weights to sample from (considering un-/labeled samples)
        scale_weights = float(len(ids_train))   # value to which weights will sum up
        sample_weight_labeled = prob_labeled * scale_weights / float(len(ids_train_labeled))
        sample_weight_unlabeled = prob_unlabeled * scale_weights \
                                    / float(len(ids_train_unlabeled)) \
                                    if len(ids_train_unlabeled) > 0 else 0.0
        sampling_weights = np.zeros((args_data.num_all_samples_train))
        sampling_weights[ids_train_labeled] = sample_weight_labeled
        sampling_weights[ids_train_unlabeled] = sample_weight_unlabeled
        num_samples_used_for_train = np.count_nonzero(sampling_weights)
        
        loader = torch.utils.data.DataLoader(
            NyuHandPoseMultiViewDataset(args_data.nyu_data_basepath, train=True, 
                                        cropSize=args_data.in_crop_size,
                                        doJitterCom=args_data.do_jitter_com,
                                        sigmaCom=args_data.sigma_com,
                                        doAddWhiteNoise=args_data.do_add_white_noise,
                                        sigmaNoise=args_data.sigma_noise,
                                        unlabeledSampleIds=ids_train_unlabeled,
                                        transform=transforms.ToTensor(),
                                        useCache=args_data.use_pickled_cache,
                                        cacheDir=args_data.nyu_data_basepath_pickled, 
                                        annoType=args_data.anno_type,
                                        neededCamIdsReal=cam_ids_corr,
                                        neededCamIdsSynth=cam_ids_corr,
                                        randomSeed=args_data.seed,
                                        cropSize3D=args_data.crop_size_3d_tuple,
                                        args_data=args_data),
            batch_size=batch_size, 
            sampler=smpl.WeightedRandomSampler(sampling_weights, 
                                               num_samples=num_samples_used_for_train, 
                                               replacement=True),
            **kwargs)
                    
        print("Using {} samples for training".format(num_samples_used_for_train))
        if sample_weight_labeled > 0.:
            print("  {} labeled".format(len(ids_train_labeled)))
        if sample_weight_unlabeled > 0.:
            print("  {} unlabeled".format(len(ids_train_unlabeled)))
            
    return loader, ids_train_permuted
            
            
def create_loader_synth_data(loader_type, args_data, do_use_gpu, 
                             num_loader_workers, batch_size, seed):
    # Sanity check
    if not loader_type == LoaderMode.TRAIN:
        print("Required loader-type {} not implemented.".format(loader_type))
        raise UserWarning("requested loader generation currently only implemented for TRAINING data loaders")
        
    return create_independent_data_loader(loader_type, args_data, do_use_gpu, 
                        num_loader_workers,
                        num_labeled_samples=np.inf, 
                        min_samp_prob=args_data.min_samp_prob, 
                        batch_size=batch_size, 
                        cam_ids_real=[], 
                        cam_ids_synth=args_data.cam_ids_for_pose_train_synth, 
                        seed=seed)
            
            
def create_independent_data_loader(loader_type, args_data, do_use_gpu, 
                                   num_loader_workers,
                                   num_labeled_samples, min_samp_prob, batch_size, 
                                   cam_ids_real, cam_ids_synth, 
                                   seed,
                                   ids_train_permuted=[]):
    """
    Creates a data loader which ignores consider correspondences between 
    samples, e.g., from different views or between real and synthetic samples.
    That is, it treads each sample completely independent.
    """
    # Sanity check
    if not loader_type == LoaderMode.TRAIN:
        print("Required loader-type {} not implemented.".format(loader_type))
        raise UserWarning("requested loader generation currently only implemented for TRAINING data loaders")
        
    kwargs = {'num_workers': num_loader_workers, 'pin_memory': True} if do_use_gpu else {}
    
    if args_data.dataset_type == DatasetType.NYU: 
        if num_labeled_samples == np.inf:
            num_labeled_samples = 100000    # value higher than #samples for this dataset
        # Set up sample IDs to sample from
        ids_train = np.arange(args_data.id_start_train, args_data.id_end_train+1)
        if len(ids_train_permuted) == 0:
            ids_train_permuted = args_data.rng.permutation(ids_train)
        ids_train_labeled = ids_train_permuted[:num_labeled_samples]
        ids_train_unlabeled = ids_train_permuted[num_labeled_samples:]
        # Ensure a minimum sampling probability for labeled samples
        ratio_labeled = len(ids_train_labeled) / float(len(ids_train))
        prob_labeled = max(min_samp_prob, ratio_labeled)
        prob_unlabeled = 1.0 - prob_labeled
        # Set up distribution/weights to sample from (considering un-/labeled samples)
        scale_weights = float(len(ids_train))   # value to which weights will sum up
        sample_weight_labeled   = prob_labeled * scale_weights \
                                    / float(len(ids_train_labeled)) \
                                    if len(ids_train_labeled) > 0 else 0.0
        sample_weight_unlabeled = prob_unlabeled * scale_weights \
                                    / float(len(ids_train_unlabeled)) \
                                    if len(ids_train_unlabeled) > 0 else 0.0
        sampling_weights = np.zeros((args_data.num_all_samples_train))
        sampling_weights[ids_train_labeled] = sample_weight_labeled
        sampling_weights[ids_train_unlabeled] = sample_weight_unlabeled
        num_samples_used_for_train = np.count_nonzero(sampling_weights)
        
        loader = torch.utils.data.DataLoader(
            NyuHandPoseDataset(args_data.nyu_data_basepath, train=True, 
                                        cropSize=args_data.in_crop_size,
                                        doJitterCom=args_data.do_jitter_com,
                                        sigmaCom=args_data.sigma_com,
                                        doAddWhiteNoise=args_data.do_add_white_noise,
                                        sigmaNoise=args_data.sigma_noise,
                                        unlabeledSampleIds=ids_train_unlabeled,
                                        transform=transforms.ToTensor(),
                                        useCache=args_data.use_pickled_cache,
                                        cacheDir=args_data.nyu_data_basepath_pickled, 
                                        annoType=args_data.anno_type,
                                        camIdsReal=cam_ids_real,
                                        camIdsSynth=cam_ids_synth,
                                        randomSeed=seed,
                                        cropSize3D=args_data.crop_size_3d_tuple,
                                        args_data=args_data),
            batch_size=batch_size, 
            sampler=smpl.WeightedRandomSampler(sampling_weights, 
                                               num_samples=num_samples_used_for_train, 
                                               replacement=True),
            **kwargs)
                    
        num_samples = num_samples_used_for_train * (len(cam_ids_real) + len(cam_ids_synth))
        print("Using {} samples for training".format(num_samples))
        if sample_weight_labeled > 0.:
            num_samples = len(ids_train_labeled) * (len(cam_ids_real) + len(cam_ids_synth))
            print("  {} labeled".format(num_samples))
        if sample_weight_unlabeled > 0.:
            num_samples = len(ids_train_unlabeled) * (len(cam_ids_real) + len(cam_ids_synth))
            print("  {} unlabeled".format(num_samples))
            
    return loader
            
            
def create_loader_preview(loader_type, args_data, do_use_gpu, 
                          num_loader_workers, min_samp_prob_label, batch_size,
                          cam_ids_real, cam_ids_synth, seed, 
                          id_range=[],
                          ids_train_permuted=[]):
    # Sanity check
    if not loader_type == LoaderMode.TRAIN:
        print("Required loader-type {} not implemented.".format(loader_type))
        raise UserWarning("requested loader generation currently only implemented for TRAINING data loaders")
        
    kwargs = {'num_workers': num_loader_workers, 'pin_memory': True} if do_use_gpu else {}
        
    if args_data.dataset_type == DatasetType.NYU:        
        if len(id_range) == 0:
            id_range = [args_data.id_start_train, args_data.id_end_train+1]
            
        # Set up sample IDs to sample from
        ids_train = np.arange(*id_range)
        if len(ids_train_permuted) == 0:
            ids_train_permuted = args_data.rng.permutation(ids_train)
        ids_train_labeled = ids_train_permuted[:args_data.num_labeled_samples]
        ids_train_unlabeled = ids_train_permuted[args_data.num_labeled_samples:]
        # Ensure a minimum sampling probability for labeled samples
        ratio_labeled = len(ids_train_labeled) / float(len(ids_train))
        prob_labeled = max(min_samp_prob_label, ratio_labeled)
        prob_unlabeled = 1.0 - prob_labeled
        # Set up distribution/weights to sample from (considering un-/labeled samples)
        scale_weights = float(len(ids_train))   # value to which weights will sum up
        sample_weight_labeled = prob_labeled * scale_weights / float(len(ids_train_labeled))
        sample_weight_unlabeled = prob_unlabeled * scale_weights \
                                    / float(len(ids_train_unlabeled)) \
                                    if len(ids_train_unlabeled) > 0 else 0.0
        sampling_weights = np.zeros((args_data.num_all_samples_train))
        sampling_weights[ids_train_labeled] = sample_weight_labeled
        sampling_weights[ids_train_unlabeled] = sample_weight_unlabeled
        num_samples_used_for_train = np.count_nonzero(sampling_weights)
        
        loader = torch.utils.data.DataLoader(
            NyuHandPoseMultiViewDataset(args_data.nyu_data_basepath, train=True, 
                                        cropSize=args_data.in_crop_size,
                                        doJitterCom=args_data.do_jitter_com,
                                        sigmaCom=args_data.sigma_com,
                                        doAddWhiteNoise=args_data.do_add_white_noise,
                                        sigmaNoise=args_data.sigma_noise,
                                        unlabeledSampleIds=ids_train_unlabeled,
                                        transform=transforms.ToTensor(),
                                        useCache=args_data.use_pickled_cache,
                                        cacheDir=args_data.nyu_data_basepath_pickled, 
                                        annoType=args_data.anno_type,
                                        neededCamIdsReal=cam_ids_real,
                                        neededCamIdsSynth=cam_ids_synth,
                                        randomSeed=seed,
                                        cropSize3D=args_data.crop_size_3d_tuple,
                                        args_data=args_data),
            batch_size=batch_size, 
            sampler=smpl.WeightedRandomSampler(sampling_weights, 
                                               num_samples=num_samples_used_for_train, 
                                               replacement=True),
            **kwargs)
                    
        print("Using {} samples for training".format(num_samples_used_for_train))
        if sample_weight_labeled > 0.:
            print("  {} labeled".format(len(ids_train_labeled)))
        if sample_weight_unlabeled > 0.:
            print("  {} unlabeled".format(len(ids_train_unlabeled)))
            
    return loader
    
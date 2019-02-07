"""
Copyright 2015, 2018 ICG, Graz University of Technology

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

# Project
from detector.handdetector import HandDetectorICG
from util.transformations import transformPoint2D, rotateImageAndGt, \
    points3DToImg_NYU, pointsImgTo3D_NYU, pointImgTo3D_NYU
from data.basetypes import Camera, NamedImgSequence, ICVLFrame, Jitter

# PyTorch
import torch
from torch.utils.data.dataset import Dataset

# General
from PIL import Image
import numpy as np
import os.path
import cPickle
import gzip
import scipy.io
from enum import IntEnum
import copy


#%% General definitions for NYU dataset
class NyuAnnoType(IntEnum):
    ALL_JOINTS = 36             # all 36 joints
    EVAL_JOINTS_ORIGINAL = 14   # original 14 evaluation joints
    
nyuRestrictedJointsEval = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
        
        
class NyuHandPoseMultiViewDataset(Dataset):
    """
    """
    
    def __init__(self, basepath, 
                 train=True, 
                 cropSize=(128, 128),
                 doJitterCom=[False, False, False],
                 sigmaCom=[20., 20., 20.],
                 doAddWhiteNoise=[False, False, False],
                 sigmaNoise=[10., 10., 10.],
                 unlabeledSampleIds=None,
                 transform=None, 
                 targetTransform=None, 
                 useCache=True, 
                 cacheDir='./cache/single_samples',
                 annoType=0,
                 neededCamIdsReal=[1, 2, 3],
                 neededCamIdsSynth=[1, 2, 3],
                 randomSeed = 123456789,
                 cropSize3D=(250,250,250),
                 args_data=None):
        """
        Initialize the dataset
        
        Arguments:
            basepath (string): base path, containing sub-folders "train" and "test"
            train (boolean, optional): True (default): use train set; 
                False: use test set
            cropSize (2-tuple, optional): size of cropped patches in pixels;
                default: 128x128
            doJitterCom (boolean, optional): 3-element list; one for each cam; 
                default = [False, False, False]
            sigmaCom (float, optional): sigma for center of mass samples 
                (in millimeters); 3-element list; one for each cam; 
                only relevant if doJitterCom is True
            doAddWhiteNoise (boolean, optional):  3-element list; one for 
                each cam; add normal distributed noise to the depth image?; 
                default: False
            sigmaNoise (float, optional): sigma for additive noise; 
                3-element list; one for each cam; 
                only relevant if doAddWhiteNoise is True
            unlabeledSampleIds (list, optional): list of sample-IDs for 
                which the label should NOT be used;
                default: None means for all samples the labels are used
            transform/targetTransform (torchvision.transforms, optional) 
                The code currently 
                assumes torchvision.transforms.ToTensor() (= default),
                better check the code if you want to add/change something
            useCache (boolean, optional): True (default): store in/load from pickle file
            cacheDir (string, optional): path to store/load pickle
            annoType (NyuAnnoType, optional): Type of annotation, i.e., 
                which joints are used.
            neededCamIdsReal (list, optional): list of camera IDs (\in {1,2,3}),
                for which real samples should be loaded (can save loading time 
                if not all cameras are needed)
            neededCamIdsSynth (list, optional): list of camera IDs (\in {1,2,3}),
                for which synthetic samples should be loaded (can save loading time 
                if not all cameras are needed)
            randomSeed (int, optional): seed for random number generator, 
                e.g., used for jittering, default: 123456789
            cropSize3D (tuple, optional): metric crop size in mm, 
                default: (250,250,250)
            args_data: object containing all parameters
        """
        print("Init NYU Dataset...")
        
        # Sanity checks
        if (not type(doJitterCom) == list) \
                or (not ((type(sigmaCom) == list) or (type(sigmaCom) == np.ndarray))):
            raise UserWarning("Parameter 'doJitterCom'/'sigmaCom' \
                must be given in a list (for each camera).")
        if (not len(doJitterCom) == 3) or (not len(sigmaCom) == 3):
            raise UserWarning("Parameters 'doJitterComnumSamples'/'sigmaCom' \
                must be 3-element lists.")
        if (not type(doAddWhiteNoise) == list) or (not type(sigmaNoise) == list):
            raise UserWarning("Parameter 'doAddWhiteNoise'/'sigmaNoise' \
                must be given in a list (for each camera).")
        if (not len(doAddWhiteNoise) == 3) or (not len(sigmaNoise) == 3):
            raise UserWarning("Parameters 'doAddWhiteNoise'/'sigmaNoise' \
                must be 3-element lists.")
                
        self.min_depth_cam = 50.
        self.max_depth_cam = 1500.

        self.args_data = args_data
        self.doTrain = train
        
        self.rng = np.random.RandomState(randomSeed)
        
        self.basepath = basepath
        # Same parameters for all three cameras (seem to be used by Tompson), 
        # and is assumed here when swaping cameras 2 and 3
        self.cam1 = Camera(camid=1, fx=588.03, fy=587.07, ux=320., uy=240.)
        self.cam2 = Camera(camid=2, fx=588.03, fy=587.07, ux=320., uy=240.)
        self.cam3 = Camera(camid=3, fx=588.03, fy=587.07, ux=320., uy=240.)
        self.cams = [self.cam1, self.cam2, self.cam3]
        self.doLoadCam1Real = 1 in neededCamIdsReal
        self.doLoadCam2Real = 2 in neededCamIdsReal
        self.doLoadCam3Real = 3 in neededCamIdsReal
        self.doLoadCam1Synth = 1 in neededCamIdsSynth
        self.doLoadCam2Synth = 2 in neededCamIdsSynth
        self.doLoadCam3Synth = 3 in neededCamIdsSynth
        self.setup_swap_id = 0
        if self.doTrain:
            self.setup_swap_id = self.args_data.setup_swap_id_train
            
        self.useCache = useCache
        self.cacheBaseDir = cacheDir
        self.restrictedJointsEval = nyuRestrictedJointsEval
        self.config = {'cube':cropSize3D}
        # For comparisons check results with adapted cube size
        scale = 0.833   # DeepPrior uses ratio 0.83 (25/30)
        self.config2 = {'cube':(cropSize3D[0]*scale, cropSize3D[1]*scale, cropSize3D[2]*scale)}
        self.testseq2_start_id = 2441
        self.cropSize = cropSize
        self.doJitterCom = doJitterCom
        self.doAddWhiteNoise = doAddWhiteNoise
        self.sigmaNoise = sigmaNoise
        self.sigmaCom = sigmaCom
        
        self.doNormZeroOne = self.args_data.do_norm_zero_one  # [-1,1] or [0,1]
        
        self.transform = transform
        self.targetTransform = targetTransform
        
        self.seqName = ""
        if self.doTrain:
            self.seqName = "train"
        else:
            self.seqName = "test"
            
        self.annoType = annoType
        self.doUseAllJoints = True
        if self.annoType == NyuAnnoType.EVAL_JOINTS_ORIGINAL:
            self.doUseAllJoints = False
            
        self.numJoints = annoType
        
        # Load labels
        trainlabels = '{}/{}/joint_data.mat'.format(basepath, self.seqName)
        self.labelMat = scipy.io.loadmat(trainlabels)
        
        # Get number of samples from annotations (test: 8252; train: 72757)
        numAllSamples = self.labelMat['joint_xyz'][self.cam1.camid-1].shape[0]
        self.numSamples = numAllSamples

        self.isSampleLabeled = np.ones((self.numSamples), dtype=bool)
        if not unlabeledSampleIds is None:
            self.isSampleLabeled[unlabeledSampleIds] = False
        
        # Assemble and create cache dir(s) if necessary
        self.cacheDirCam1Real,self.cacheDirCam2Real,self.cacheDirCam3Real = "","",""
        self.cacheDirCam1Synth,self.cacheDirCam2Synth,self.cacheDirCam3Synth = "","",""
        if self.useCache:
            # Real data
            realString = "real"
            # Cam1
            camString = "cam{}".format(self.cam1.camid)
            self.cacheDirCam1Real = os.path.join(
                self.cacheBaseDir, self.seqName, realString, camString)
            if not os.path.exists(self.cacheDirCam1Real):
                os.makedirs(self.cacheDirCam1Real)
            # Cam2
            camString = "cam{}".format(self.cam2.camid)
            self.cacheDirCam2Real = os.path.join(
                self.cacheBaseDir, self.seqName, realString, camString)
            if not os.path.exists(self.cacheDirCam2Real):
                os.makedirs(self.cacheDirCam2Real)
            # Cam3
            camString = "cam{}".format(self.cam3.camid)
            self.cacheDirCam3Real = os.path.join(
                self.cacheBaseDir, self.seqName, realString, camString)
            if not os.path.exists(self.cacheDirCam3Real):
                os.makedirs(self.cacheDirCam3Real)
            # Synthetic data
            synthString = "synthetic"
            # Cam1
            camString = "cam{}".format(self.cam1.camid)
            self.cacheDirCam1Synth = os.path.join(
                self.cacheBaseDir, self.seqName, synthString, camString)
            if not os.path.exists(self.cacheDirCam1Synth):
                os.makedirs(self.cacheDirCam1Synth)
            # Cam2
            camString = "cam{}".format(self.cam2.camid)
            self.cacheDirCam2Synth = os.path.join(
                self.cacheBaseDir, self.seqName, synthString, camString)
            if not os.path.exists(self.cacheDirCam2Synth):
                os.makedirs(self.cacheDirCam2Synth)
            # Cam3
            camString = "cam{}".format(self.cam3.camid)
            self.cacheDirCam3Synth = os.path.join(
                self.cacheBaseDir, self.seqName, synthString, camString)
            if not os.path.exists(self.cacheDirCam3Synth):
                os.makedirs(self.cacheDirCam3Synth)
                
        # Precomputations for normalization of 3D point
        self.precompute_normalization_factors()
        
        print("NYU Dataset init done.")


    def __getitem__(self, index):
        # Cam1
        cam_id = 1
        img_c1_r, target_c1_r, transform_crop_c1_r, com_c1_r, size_c1_r = [], [], [], [], []
        jitter_c1 = None
        if self.doLoadCam1Real:
            is_real = True
            img_c1_r, target_c1_r, transform_crop_c1_r, com_c1_r, size_c1_r, jitter_c1 \
                = self.load_and_prepare_sample(index, is_real, cam_id)
                
        img_c1_s, target_c1_s, transform_crop_c1_s, com_c1_s, size_c1_s = [], [], [], [], []
        if self.doLoadCam1Synth:
            is_real = False
            img_c1_s, target_c1_s, transform_crop_c1_s, com_c1_s, size_c1_s, _ \
                = self.load_and_prepare_sample(index, is_real, cam_id, fixedJitter=jitter_c1)
                
        # Cam2
        cam_id = 2
        if index < self.setup_swap_id:
            cam_id = 3
        img_c2_r, target_c2_r, transform_crop_c2_r, com_c2_r, size_c2_r = [], [], [], [], []
        jitter_c2 = None
        if self.doLoadCam2Real:
            is_real = True
            img_c2_r, target_c2_r, transform_crop_c2_r, com_c2_r, size_c2_r, jitter_c2 \
                = self.load_and_prepare_sample(index, is_real, cam_id)
                
        img_c2_s, target_c2_s, transform_crop_c2_s, com_c2_s, size_c2_s = [], [], [], [], []
        if self.doLoadCam2Synth:
            is_real = False
            img_c2_s, target_c2_s, transform_crop_c2_s, com_c2_s, size_c2_s, _ \
                = self.load_and_prepare_sample(index, is_real, cam_id, fixedJitter=jitter_c2)
                
        # Cam3
        cam_id = 3
        if index < self.setup_swap_id:
            cam_id = 2
        img_c3_r, target_c3_r, transform_crop_c3_r, com_c3_r, size_c3_r = [], [], [], [], []
        jitter_c3 = None
        if self.doLoadCam3Real:
            is_real = True
            img_c3_r, target_c3_r, transform_crop_c3_r, com_c3_r, size_c3_r, jitter_c3 \
                = self.load_and_prepare_sample(index, is_real, cam_id)
                
        img_c3_s, target_c3_s, transform_crop_c3_s, com_c3_s, size_c3_s = [], [], [], [], []
        if self.doLoadCam3Synth:
            is_real = False
            img_c3_s, target_c3_s, transform_crop_c3_s, com_c3_s, size_c3_s, _ \
                = self.load_and_prepare_sample(index, is_real, cam_id, fixedJitter=jitter_c3)
                
        is_labeled = 1 if self.isSampleLabeled[index] else 0
            
        # c1 = cam1, c2 = ..., r = real, s = synthetic
        return img_c1_r, img_c2_r, img_c3_r, target_c1_r, target_c2_r, target_c3_r, \
            transform_crop_c1_r, transform_crop_c2_r, transform_crop_c3_r, \
            com_c1_r, com_c2_r, com_c3_r, \
            size_c1_r, size_c2_r, size_c3_r, \
            img_c1_s, img_c2_s, img_c3_s, target_c1_s, target_c2_s, target_c3_s, \
            transform_crop_c1_s, transform_crop_c2_s, transform_crop_c3_s, \
            com_c1_s, com_c2_s, com_c3_s, \
            size_c1_s, size_c2_s, size_c3_s, \
            torch.ByteTensor([is_labeled])


    def __len__(self):
        return self.numSamples
        
        
    def load_and_prepare_sample(self, index, is_real, cam_id, fixedJitter=None):
        """
        Load a single data sample
        
        Arguments:
            index (int): Sample index
            is_real (boolean): whether to load a real sample, otherwise synthetic
            cam_id (int): 1-based camera ID, \in {1,2,3}
            fixedJitter (Jitter, optional): jitter definition, i.e., random 
                samples to be used, default: None, means jitter is randomly 
                sampled, i.e., not fixed
        """
        # Set some camera, real/synth specific parameters
        if cam_id == 1:
            fx, fy, cx, cy = self.cam1.fx, self.cam1.fy, self.cam1.ux, self.cam1.uy
            if is_real:
                cache_dir = self.cacheDirCam1Real
            else:
                cache_dir = self.cacheDirCam1Synth
        if cam_id == 2:
            fx, fy, cx, cy = self.cam2.fx, self.cam2.fy, self.cam2.ux, self.cam2.uy
            if is_real:
                cache_dir = self.cacheDirCam2Real
            else:
                cache_dir = self.cacheDirCam2Synth
        if cam_id == 3:
            fx, fy, cx, cy = self.cam3.fx, self.cam3.fy, self.cam3.ux, self.cam3.uy
            if is_real:
                cache_dir = self.cacheDirCam3Real
            else:
                cache_dir = self.cacheDirCam3Synth
        # Set train/test specific parameters
        config = self.config
        if self.doTrain:
            doJitterCubesize = self.args_data.do_jitter_cubesize[cam_id-1]
            doJitterRotation = self.args_data.do_jitter_rotation[cam_id-1]
        else:
            doJitterCubesize = self.args_data.do_jitter_cubesize_test[cam_id-1]
            doJitterRotation = self.args_data.do_jitter_rotation_test[cam_id-1]
            if index >= self.testseq2_start_id:
                config = self.config2   # for comparison to some rel. work
        # Synthetic data is always labeled
        is_sample_labeled = self.isSampleLabeled[index] if is_real else True
            
        dataSeq, jitter = loadSingleSampleNyu(self.basepath, self.seqName, index, 
                                   self.rng,
                                   doLoadRealSample=is_real,
                                   camId=cam_id, 
                                   fx=fx, fy=fy, ux=cx, uy=cy,
                                   allJoints=self.doUseAllJoints, 
                                   config=config,
                                   cropSize=self.cropSize,
                                   doJitterCom=self.doJitterCom[cam_id-1],
                                   sigmaCom=self.sigmaCom[cam_id-1],
                                   doAddWhiteNoise=self.doAddWhiteNoise[cam_id-1],
                                   sigmaNoise=self.sigmaNoise[cam_id-1],
                                   doJitterCubesize=doJitterCubesize,
                                   sigmaCubesize=self.args_data.sigma_cubesize[cam_id-1],
                                   doJitterRotation=doJitterRotation,
                                   rotationAngleRange=self.args_data.rotation_angle_range,
                                   useCache=self.useCache,
                                   cacheDir=cache_dir,
                                   labelMat=self.labelMat,
                                   doUseLabel=is_sample_labeled,
                                   minRatioInside=self.args_data.min_ratio_inside,
                                   jitter=fixedJitter)
                                   
        data = dataSeq.data[0]
                                        
        if self.doNormZeroOne:
            img, target = normalizeZeroOne(data)
        else:
            img, target = normalizeMinusOneOne(data)
            
        # Image need to be HxWxC and will be divided by transform (ToTensor()), which is assumed here! 
        img = np.expand_dims(img, axis=2)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.targetTransform is not None:
            target = self.targetTransform(target)
            
        target = torch.from_numpy(target.astype('float32'))
        transform_crop = torch.from_numpy(data.T)
        com = torch.from_numpy(data.com)
        cubesize = torch.Tensor([data.config['cube'][0], data.config['cube'][1], data.config['cube'][2]])
        
        return img, target, transform_crop, com, cubesize, jitter
        

    def get_type_string(self):
        return "NYU"
        
        
    def denormalize_joint_pos(self, jointPos, cubesize):
        """
        Re-scale the given joint positions to metric distances (in mm)
        
        Arguments:
            jointPos (numpy.ndarray): normalized joint positions, 
                as provided by the dataset
            cubesize: hand cube size
        """
        return denormalizeJointPositions(jointPos, self.doNormZeroOne, cubesize)
        
        
    def precompute_normalization_factors(self):
        min_depth_in = self.min_depth_cam
        max_depth_in = self.max_depth_cam
        
        if self.args_data.do_norm_zero_one:
            depth_range_out = 1.
            self.norm_min_out = 0.
        else:
            depth_range_out = 2.
            self.norm_min_out = -1.
            
        depth_range_in = float(max_depth_in - min_depth_in)
        
        self.norm_max_out = 1.
        self.norm_min_in = min_depth_in
        self.norm_scale_3Dpt_2_norm = depth_range_out / depth_range_in

    
    def normalize_3D(self, points_3D):
        """
        Normalize depth to a desired range; x and y are normalized accordingly
        range for x,y is double the depth range 
        This essentially assumes only positive z, but pos/neg x and y as input
        
        Arguments:
            points_3D (Nx3 numpy array): array of N 3D points
        """
        pt = np.asarray(copy.deepcopy(points_3D), 'float32')
        
        pt[:,0] = pt[:,0] * self.norm_scale_3Dpt_2_norm
        pt[:,1] = pt[:,1] * self.norm_scale_3Dpt_2_norm
        pt[:,2] = (pt[:,2] - self.norm_min_in) * self.norm_scale_3Dpt_2_norm + self.norm_min_out
        
        np.clip(pt, self.norm_min_out, self.norm_max_out, out=pt)
        
        return pt

    
    def normalize_and_jitter_3D(self, points_3D):
        """
        Normalize depth to a desired range; x and y are normalized accordingly
        range for x,y is double the depth range 
        This essentially assumes only positive z, but pos/neg x and y as input
        Additionally noise is added
        
        Arguments:
            points_3D (Nx3 numpy array): array of N 3D points
        """
        pt = np.asarray(copy.deepcopy(points_3D), 'float32')
        
        # Add noise to original 3D coords (in mm)
        sigma = 15.
        pt += sigma * self.args_data.rng.randn(pt.shape[0], pt.shape[1])
        
        pt[:,0] = pt[:,0] * self.norm_scale_3Dpt_2_norm
        pt[:,1] = pt[:,1] * self.norm_scale_3Dpt_2_norm
        pt[:,2] = (pt[:,2] - self.norm_min_in) * self.norm_scale_3Dpt_2_norm + self.norm_min_out
        
        np.clip(pt, self.norm_min_out, self.norm_max_out, out=pt)
        
        return pt
        
        
    def points3DToImg(self, points, cam_id=1):
        """
        Arguments:
            points
            cam_id (int, optional): 1-based camera ID in {1,2,3} 
                to select the calibration parameters, default: 1
        """
        cam = self.cams[cam_id-1]
        return points3DToImg_NYU(points, cam.fx, cam.fy, cam.ux, cam.uy)
        
        
class NyuHandPoseDataset(Dataset):
    """
    Dataset for sampling single samples without considering correspondences
    between views or real and synthetic data, i.e., considering each sample 
    completely independent.
    """
    
    def __init__(self, basepath,
                 train=True, 
                 cropSize=(128, 128),
                 doJitterCom=[False, False, False],
                 sigmaCom=[20., 20., 20.],
                 doAddWhiteNoise=[False, False, False],
                 sigmaNoise=[10., 10., 10.],
                 unlabeledSampleIds=None,
                 transform=None, 
                 targetTransform=None, 
                 useCache=True, 
                 cacheDir='./cache/single_samples',
                 annoType=0,
                 camIdsReal=[1, 2, 3],
                 camIdsSynth=[1, 2, 3],
                 randomSeed = 123456789,
                 cropSize3D=(250,250,250),
                 args_data=None): 
        """
        Initialize the dataset
        
        Arguments:
            basepath (string): base path, containing sub-folders "train" and "test"
            train (boolean, optional): True (default): use train set; 
                False: use test set
            cropSize (2-tuple, optional): size of cropped patches in pixels;
                default: 128x128
            doJitterCom (boolean, optional): 3-element list; one for each cam; 
                default = [False, False, False]
            sigmaCom (float, optional): sigma for center of mass samples 
                (in millimeters); 3-element list; one for each cam; 
                only relevant if doJitterCom is True
            doAddWhiteNoise (boolean, optional):  3-element list; one for 
                each cam; add normal distributed noise to the depth image?; 
                default: False
            sigmaNoise (float, optional): sigma for additive noise; 
                3-element list; one for each cam; 
                only relevant if doAddWhiteNoise is True
            unlabeledSampleIds (list, optional): list of sample-IDs for 
                which the label should NOT be used;
                default: None means for all samples the labels are used
            transform/targetTransform (torchvision.transforms, optional) 
                The code currently 
                assumes torchvision.transforms.ToTensor() (= default),
                better check the code if you want to add/change something
            useCache (boolean, optional): True (default): store in/load from pickle file
            cacheDir (string, optional): path to store/load pickle
            annoType (NyuAnnoType, optional): Type of annotation, i.e., 
                which joints are used.
            camIdsReal (list, optional): list of camera IDs (\in {1,2,3}),
                from which real samples can be loaded
            camIdsSynth (list, optional): list of camera IDs (\in {1,2,3}),
                for which synthetic samples can be loaded
            randomSeed (int, optional): seed for random number generator, 
                e.g., used for jittering, default: 123456789
            cropSize3D (tuple, optional): metric crop size in mm, 
                default: (250,250,250)
            args_data: object containing all parameters
        """
        print("Init NYU Dataset...")
        
        # Sanity checks
        if (not type(doJitterCom) == list) \
                or (not ((type(sigmaCom) == list) or (type(sigmaCom) == np.ndarray))):
            raise UserWarning("Parameter 'doJitterCom'/'sigmaCom' \
                must be given in a list (for each camera).")
        if (not len(doJitterCom) == 3) or (not len(sigmaCom) == 3):
            raise UserWarning("Parameters 'doJitterComnumSamples'/'sigmaCom' \
                must be 3-element lists.")
        if (not type(doAddWhiteNoise) == list) or (not type(sigmaNoise) == list):
            raise UserWarning("Parameter 'doAddWhiteNoise'/'sigmaNoise' \
                must be given in a list (for each camera).")
        if (not len(doAddWhiteNoise) == 3) or (not len(sigmaNoise) == 3):
            raise UserWarning("Parameters 'doAddWhiteNoise'/'sigmaNoise' \
                must be 3-element lists.")
                
        self.min_depth_cam = 50.
        self.max_depth_cam = 1500.

        self.args_data = args_data
        
        self.rng = np.random.RandomState(randomSeed)
        
        self.basepath = basepath
        # Same parameters for all three cameras (seem to be used by Tompson)
        self.cam1 = Camera(camid=1, fx=588.03, fy=587.07, ux=320., uy=240.)
        self.cam2 = Camera(camid=2, fx=588.03, fy=587.07, ux=320., uy=240.)
        self.cam3 = Camera(camid=3, fx=588.03, fy=587.07, ux=320., uy=240.)
        self.cams = [self.cam1, self.cam2, self.cam3]
        self.camIdsReal = camIdsReal
        self.camIdsSynth = camIdsSynth
        self.useCache = useCache
        self.cacheBaseDir = cacheDir
        self.restrictedJointsEval = nyuRestrictedJointsEval
        self.config = {'cube':cropSize3D}
        # For comparisons check results with adapted cube size
        scale = 0.833   # DeepPrior uses ratio 0.83 (25/30)
        self.config2 = {'cube':(cropSize3D[0]*scale, cropSize3D[1]*scale, cropSize3D[2]*scale)}
        self.testseq2_start_id = 2441
        self.cropSize = cropSize
        self.doJitterCom = doJitterCom
        self.doAddWhiteNoise = doAddWhiteNoise
        self.sigmaNoise = sigmaNoise
        self.sigmaCom = sigmaCom
        
        self.doNormZeroOne = self.args_data.do_norm_zero_one  # [-1,1] or [0,1]
        
        self.transform = transform
        self.targetTransform = targetTransform
        
        self.doTrain = train
        self.seqName = ""
        if self.doTrain:
            self.seqName = "train"
        else:
            self.seqName = "test"
            
        self.annoType = annoType
        self.doUseAllJoints = True
        if self.annoType == NyuAnnoType.EVAL_JOINTS_ORIGINAL:
            self.doUseAllJoints = False
            
        self.numJoints = annoType
        
        # Load labels
        trainlabels = '{}/{}/joint_data.mat'.format(basepath, self.seqName)
        self.labelMat = scipy.io.loadmat(trainlabels)
        
        # Get number of samples from annotations (test: 8252; train: 72757)
        numAllSamples = self.labelMat['joint_xyz'][self.cam1.camid-1].shape[0]
        self.numSamples = numAllSamples
        
        self.isSampleLabeled = np.ones((self.numSamples), dtype=bool)
        if not unlabeledSampleIds is None:
            self.isSampleLabeled[unlabeledSampleIds] = False

        # Assemble list from which to sample whether a real or synthetic data sample shall be loaded
        self.real_synth_sample_list = []
        if len(self.camIdsSynth) > 0:
            self.real_synth_sample_list.append(0)
        if len(self.camIdsReal) > 0:
            self.real_synth_sample_list.append(1)
        
        # Assemble and create cache dir(s) if necessary
        self.cacheDirCam1Real,self.cacheDirCam2Real,self.cacheDirCam3Real = "","",""
        self.cacheDirCam1Synth,self.cacheDirCam2Synth,self.cacheDirCam3Synth = "","",""
        if self.useCache:
            # Real data
            realString = "real"
            if 1 in self.camIdsReal:
                camString = "cam{}".format(self.cam1.camid)
                self.cacheDirCam1Real = os.path.join(
                    self.cacheBaseDir, self.seqName, realString, camString)
                if not os.path.exists(self.cacheDirCam1Real):
                    os.makedirs(self.cacheDirCam1Real)
            if 2 in self.camIdsReal:
                camString = "cam{}".format(self.cam2.camid)
                self.cacheDirCam2Real = os.path.join(
                    self.cacheBaseDir, self.seqName, realString, camString)
                if not os.path.exists(self.cacheDirCam2Real):
                    os.makedirs(self.cacheDirCam2Real)
            if 3 in self.camIdsReal:
                camString = "cam{}".format(self.cam3.camid)
                self.cacheDirCam3Real = os.path.join(
                    self.cacheBaseDir, self.seqName, realString, camString)
                if not os.path.exists(self.cacheDirCam3Real):
                    os.makedirs(self.cacheDirCam3Real)
            # Synthetic data
            synthString = "synthetic"
            if 1 in self.camIdsSynth:
                camString = "cam{}".format(self.cam1.camid)
                self.cacheDirCam1Synth = os.path.join(
                    self.cacheBaseDir, self.seqName, synthString, camString)
                if not os.path.exists(self.cacheDirCam1Synth):
                    os.makedirs(self.cacheDirCam1Synth)
            if 2 in self.camIdsSynth:
                camString = "cam{}".format(self.cam2.camid)
                self.cacheDirCam2Synth = os.path.join(
                    self.cacheBaseDir, self.seqName, synthString, camString)
                if not os.path.exists(self.cacheDirCam2Synth):
                    os.makedirs(self.cacheDirCam2Synth)
            if 3 in self.camIdsSynth:
                camString = "cam{}".format(self.cam3.camid)
                self.cacheDirCam3Synth = os.path.join(
                    self.cacheBaseDir, self.seqName, synthString, camString)
                if not os.path.exists(self.cacheDirCam3Synth):
                    os.makedirs(self.cacheDirCam3Synth)
                
        # Precomputations for normalization of 3D point
        self.precompute_normalization_factors()
        
        print("NYU Dataset init done.")


    def __getitem__(self, index):
        # Sample real/synth
        is_real_int = self.real_synth_sample_list[self.rng.randint(len(self.real_synth_sample_list))]
        if is_real_int > 0:
            is_real = True
            num_cams = len(self.camIdsReal)
            # Sample cam ID
            cam_id = self.camIdsReal[self.rng.randint(num_cams)]
        else:
            is_real = False
            num_cams = len(self.camIdsSynth)
            # Sample cam ID
            cam_id = self.camIdsSynth[self.rng.randint(num_cams)]
        
        img, target, transform, com, size, jitter \
            = self.load_and_prepare_sample(index, is_real, cam_id)
            
        return img, target, transform, com, size


    def __len__(self):
        return self.numSamples
        
        
    def load_and_prepare_sample(self, index, is_real, cam_id, fixedJitter=None):
        """
        Load a single data sample
        
        Arguments:
            index (int): Sample index
            is_real (boolean): whether to load a real sample, otherwise synthetic
            cam_id (int): 1-based camera ID, \in {1,2,3}
            fixedJitter (Jitter, optional): jitter definition, i.e., random 
                samples to be used, default: None, means jitter is randomly 
                sampled, i.e., not fixed
        """
        # Set some camera, real/synth specific parameters
        if cam_id == 1:
            fx, fy, cx, cy = self.cam1.fx, self.cam1.fy, self.cam1.ux, self.cam1.uy
            if is_real:
                cache_dir = self.cacheDirCam1Real
            else:
                cache_dir = self.cacheDirCam1Synth
        if cam_id == 2:
            fx, fy, cx, cy = self.cam2.fx, self.cam2.fy, self.cam2.ux, self.cam2.uy
            if is_real:
                cache_dir = self.cacheDirCam2Real
            else:
                cache_dir = self.cacheDirCam2Synth
        if cam_id == 3:
            fx, fy, cx, cy = self.cam3.fx, self.cam3.fy, self.cam3.ux, self.cam3.uy
            if is_real:
                cache_dir = self.cacheDirCam3Real
            else:
                cache_dir = self.cacheDirCam3Synth
        # Set train/test specific parameters
        config = self.config
        if self.doTrain:
            doJitterCubesize = self.args_data.do_jitter_cubesize[cam_id-1]
            doJitterRotation = self.args_data.do_jitter_rotation[cam_id-1]
        else:
            doJitterCubesize = self.args_data.do_jitter_cubesize_test[cam_id-1]
            doJitterRotation = self.args_data.do_jitter_rotation_test[cam_id-1]
            if index >= self.testseq2_start_id:
                config = self.config2   # for comparison to some rel. work
        # Synthetic data is always labeled
        is_sample_labeled = self.isSampleLabeled[index] if is_real else True
            
        dataSeq, jitter = loadSingleSampleNyu(self.basepath, self.seqName, index, 
                                   self.rng,
                                   doLoadRealSample=is_real,
                                   camId=cam_id, 
                                   fx=fx, fy=fy, ux=cx, uy=cy,
                                   allJoints=self.doUseAllJoints, 
                                   config=config,
                                   cropSize=self.cropSize,
                                   doJitterCom=self.doJitterCom[cam_id-1],
                                   sigmaCom=self.sigmaCom[cam_id-1],
                                   doAddWhiteNoise=self.doAddWhiteNoise[cam_id-1],
                                   sigmaNoise=self.sigmaNoise[cam_id-1],
                                   doJitterCubesize=doJitterCubesize,
                                   sigmaCubesize=self.args_data.sigma_cubesize[cam_id-1],
                                   doJitterRotation=doJitterRotation,
                                   rotationAngleRange=self.args_data.rotation_angle_range,
                                   useCache=self.useCache,
                                   cacheDir=cache_dir,
                                   labelMat=self.labelMat,
                                   doUseLabel=is_sample_labeled,
                                   minRatioInside=self.args_data.min_ratio_inside,
                                   jitter=fixedJitter)
                                   
        data = dataSeq.data[0]
                                        
        if self.doNormZeroOne:
            img, target = normalizeZeroOne(data)
        else:
            img, target = normalizeMinusOneOne(data)
            
        # Image need to be HxWxC and will be divided by transform (ToTensor()), which is assumed here! 
        img = np.expand_dims(img, axis=2)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.targetTransform is not None:
            target = self.targetTransform(target)
            
        target = torch.from_numpy(target.astype('float32'))
        transform_crop = torch.from_numpy(data.T)
        com = torch.from_numpy(data.com)
        cubesize = torch.Tensor([data.config['cube'][0], data.config['cube'][1], data.config['cube'][2]])
        
        return img, target, transform_crop, com, cubesize, jitter
        

    def get_type_string(self):
        return "NYU"
        
        
    def denormalize_joint_pos(self, jointPos, cubesize):
        """
        Re-scale the given joint positions to metric distances (in mm)
        
        Arguments:
            jointPos (numpy.ndarray): normalized joint positions, 
                as provided by the dataset
            cubesize: hand cube size
        """
        return denormalizeJointPositions(jointPos, self.doNormZeroOne, cubesize)
        
        
    def precompute_normalization_factors(self):
        min_depth_in = self.min_depth_cam
        max_depth_in = self.max_depth_cam
        
        if self.args_data.do_norm_zero_one:
            depth_range_out = 1.
            self.norm_min_out = 0.
        else:
            depth_range_out = 2.
            self.norm_min_out = -1.
            
        depth_range_in = float(max_depth_in - min_depth_in)
        
        self.norm_max_out = 1.
        self.norm_min_in = min_depth_in
        self.norm_scale_3Dpt_2_norm = depth_range_out / depth_range_in

    
    def normalize_3D(self, points_3D):
        """
        Normalize depth to a desired range; x and y are normalized accordingly
        range for x,y is double the depth range 
        This essentially assumes only positive z, but pos/neg x and y as input
        
        Arguments:
            points_3D (Nx3 numpy array): array of N 3D points
        """
        pt = np.asarray(copy.deepcopy(points_3D), 'float32')
        
        pt[:,0] = pt[:,0] * self.norm_scale_3Dpt_2_norm
        pt[:,1] = pt[:,1] * self.norm_scale_3Dpt_2_norm
        pt[:,2] = (pt[:,2] - self.norm_min_in) * self.norm_scale_3Dpt_2_norm + self.norm_min_out
        
        np.clip(pt, self.norm_min_out, self.norm_max_out, out=pt)
        
        return pt

    
    def normalize_and_jitter_3D(self, points_3D):
        """
        Normalize depth to a desired range; x and y are normalized accordingly
        range for x,y is double the depth range 
        This essentially assumes only positive z, but pos/neg x and y as input
        Additionally noise is added
        
        Arguments:
            points_3D (Nx3 numpy array): array of N 3D points
        """
        pt = np.asarray(copy.deepcopy(points_3D), 'float32')
        
        # Add noise to original 3D coords (in mm)
        sigma = 15.
        pt += sigma * self.args_data.rng.randn(pt.shape[0], pt.shape[1])
        
        pt[:,0] = pt[:,0] * self.norm_scale_3Dpt_2_norm
        pt[:,1] = pt[:,1] * self.norm_scale_3Dpt_2_norm
        pt[:,2] = (pt[:,2] - self.norm_min_in) * self.norm_scale_3Dpt_2_norm + self.norm_min_out
        
        np.clip(pt, self.norm_min_out, self.norm_max_out, out=pt)
        
        return pt
        
        
    def points3DToImg(self, points, cam_id=1):
        """
        Arguments:
            points
            cam_id (int, optional): 1-based camera ID in {1,2,3} 
                to select the calibration parameters, default: 1
        """
        cam = self.cams[cam_id-1]
        return points3DToImg_NYU(points, cam.fx, cam.fy, cam.ux, cam.uy)


#%% Helpers

def loadDepthMap(filename):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """
    with open(filename) as f:
        img = Image.open(filename)
        # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        assert len(img.getbands()) == 3
        r, g, b = img.split()
        r = np.asarray(r,np.int32)
        g = np.asarray(g,np.int32)
        b = np.asarray(b,np.int32)
        dpt = np.bitwise_or(np.left_shift(g,8),b)
        imgdata = np.asarray(dpt,np.float32)

    return imgdata
    
    
def loadSingleSampleNyu(basepath, seqName, index, rng,
                        doLoadRealSample=True,
                        camId=1, fx=588.03, fy=587.07, ux=320., uy=240.,
                        allJoints=False, 
                        config={'cube': (250,250,250)},
                        cropSize=(128,128),
                        doJitterCom=False,
                        sigmaCom=20.,
                        doAddWhiteNoise=False,
                        sigmaNoise=10.,
                        doJitterCubesize=False,
                        sigmaCubesize=20.,
                        doJitterRotation=False,
                        rotationAngleRange=[-45.0, 45.0],
                        useCache=True,
                        cacheDir="./cache/single_samples",
                        labelMat=None,
                        doUseLabel=True,
                        minRatioInside=0.3,
                        jitter=None):
    """
    Load an image sequence from the NYU hand pose dataset
    
    Arguments:
        basepath (string): base path, containing sub-folders "train" and "test"
        seqName: sequence name, e.g. train
        index (int): index of image to be loaded
        rng (random number generator): as returned by numpy.random.RandomState
        doLoadRealSample (boolean, optional): whether to load the real 
            sample, i.e., captured by the camera (True; default) or 
            the synthetic sample, rendered using a hand model (False)
        camId (int, optional): camera ID, as used in filename; \in {1,2,3}; 
            default: 1
        fx, fy (float, optional): camera focal length; default for cam 1
        ux, uy (float, optional): camera principal point; default for cam 1
        allJoints (boolean): use all 36 joints or just the eval.-subset
        config (dictionary, optional): need to have a key 'cube' whose value 
            is a 3-tuple specifying the cube-size (x,y,z in mm) 
            which is extracted around the found hand center location
        cropSize (2-tuple, optional): size of cropped patches in pixels;
            default: 128x128
        doJitterCom (boolean, optional): default: False
        sigmaCom (float, optional): sigma for center of mass samples 
            (in millimeters); only relevant if doJitterCom is True
        sigmaNoise (float, optional): sigma for additive noise; 
            only relevant if doAddWhiteNoise is True
        doAddWhiteNoise (boolean, optional): add normal distributed noise 
            to the depth image; default: False
        useCache (boolean, optional): True (default): store in/load from 
            pickle file
        cacheDir (string, optional): path to store/load pickle
        labelMat (optional): loaded mat file; (full file need to be loaded for 
            each sample if not given)
        doUseLabel (bool, optional): specify if the label should be used;
            default: True
        
    Returns:
        named image sequence
    """
    
    idComGT = 13
    if allJoints:
        idComGT = 34
        
    if jitter == None:
        doUseFixedJitter = False
        jitter = Jitter()
    else:
        doUseFixedJitter = True
        
    # Load the dataset
    objdir = '{}/{}/'.format(basepath,seqName)

    if labelMat == None:
        trainlabels = '{}/{}/joint_data.mat'.format(basepath, seqName)
        labelMat = scipy.io.loadmat(trainlabels)
        
    joints3D = labelMat['joint_xyz'][camId-1]
    joints2D = labelMat['joint_uvd'][camId-1]
    if allJoints:
        eval_idxs = np.arange(36)
    else:
        eval_idxs = nyuRestrictedJointsEval

    numJoints = len(eval_idxs)
    
    data = []
    line = index
    
    # Assemble original filename
    prefix = "depth" if doLoadRealSample else "synthdepth"
    dptFileName = '{0:s}/{1:s}_{2:1d}_{3:07d}.png'.format(objdir, prefix, camId, line+1)
    # Assemble pickled filename
    cacheFilename = "frame_{}_all{}.pgz".format(index, allJoints)
    pickleCacheFile = os.path.join(cacheDir, cacheFilename)
        
    # Load image
    if useCache and os.path.isfile(pickleCacheFile):
        # Load from pickle file
        with gzip.open(pickleCacheFile, 'rb') as f:
            try:
                dpt = cPickle.load(f)
            except:
                print("Data file exists but failed to load. File: {}".format(pickleCacheFile))
                raise
        
    else:
        # Load from original file
        if not os.path.isfile(dptFileName):
            raise UserWarning("Desired image file from NYU dataset does not exist \
                (Filename: {}) \
                use cache? {}, cache file: {}".format(dptFileName, useCache, pickleCacheFile))
        dpt = loadDepthMap(dptFileName)
    
        # Write to pickle file
        if useCache:
            with gzip.GzipFile(pickleCacheFile, 'wb') as f:
                cPickle.dump(dpt, f, protocol=cPickle.HIGHEST_PROTOCOL)
                
    # Add noise?
    if doAddWhiteNoise:
        jitter.img_white_noise_scale = rng.randn(dpt.shape[0], dpt.shape[1])
        dpt = dpt + sigmaNoise * jitter.img_white_noise_scale
    else:
        jitter.img_white_noise_scale = np.zeros((dpt.shape[0], dpt.shape[1]), dtype=np.float32)
    
    # joints in image coordinates
    gtorig = np.zeros((numJoints, 3), np.float32)
    jt = 0
    for ii in range(joints2D.shape[1]):
        if ii not in eval_idxs:
            continue
        gtorig[jt,0] = joints2D[line,ii,0]
        gtorig[jt,1] = joints2D[line,ii,1]
        gtorig[jt,2] = joints2D[line,ii,2]
        jt += 1

    # normalized joints in 3D coordinates
    gt3Dorig = np.zeros((numJoints,3),np.float32)
    jt = 0
    for jj in range(joints3D.shape[1]):
        if jj not in eval_idxs:
            continue
        gt3Dorig[jt,0] = joints3D[line,jj,0]
        gt3Dorig[jt,1] = joints3D[line,jj,1]
        gt3Dorig[jt,2] = joints3D[line,jj,2]
        jt += 1
        
    if doJitterRotation:
        if not doUseFixedJitter:
            jitter.rotation_angle_scale = rng.rand(1)
        rot = jitter.rotation_angle_scale * (rotationAngleRange[1] - rotationAngleRange[0]) + rotationAngleRange[0]
        dpt, gtorig, gt3Dorig = rotateImageAndGt(dpt, gtorig, gt3Dorig, rot, 
                                                 fx, fy, ux, uy, 
                                                 jointIdRotCenter=idComGT, 
                                                 pointsImgTo3DFunction=pointsImgTo3D,
                                                 bgValue=10000)
    else:
        # Compute scale factor corresponding to no rotation
        jitter.rotation_angle_scale = -rotationAngleRange[0] / (rotationAngleRange[1] - rotationAngleRange[0])
        
    # Detect hand
    hdOwn = HandDetectorICG()
    comGT = copy.deepcopy(gtorig[idComGT])  # use GT position for comparison
        
    # Jitter com?
    comOffset = np.zeros((3), np.float32)
    if doJitterCom:
        if not doUseFixedJitter:
            jitter.detection_offset_scale = rng.randn(3)
        comOffset = sigmaCom * jitter.detection_offset_scale
        # Transform x/y to pixel coords (since com is in uvd coords)
        comOffset[0] = (fx * comOffset[0]) / (comGT[2] + comOffset[2])
        comOffset[1] = (fy * comOffset[1]) / (comGT[2] + comOffset[2])
    else:
        jitter.detection_offset_scale = np.zeros((3), dtype=np.float32)        
    comGT = comGT + comOffset
    
    # Jitter scale (cube size)?
    cubesize = np.array((config['cube'][0], config['cube'][1], config['cube'][2]), dtype=np.float64)
    if doJitterCubesize:
        if not doUseFixedJitter:
            jitter.crop_scale = rng.rand(1)
        # sigma defines range for uniform distr.
        cubesizeChange = 2 * sigmaCubesize * jitter.crop_scale - sigmaCubesize
        cubesize += cubesizeChange
    else:
        jitter.crop_scale = 0.5
    
    dpt, M, com = hdOwn.cropArea3D(imgDepth=dpt, com=comGT, fx=fx, fy=fy, 
                                   minRatioInside=minRatioInside, \
                                   size=cubesize, dsize=cropSize)
                                    
    com3D = jointImgTo3D(com, fx, fy, ux, uy)
    gt3Dcrop = gt3Dorig - com3D     # normalize to com
    gtcrop = np.zeros((gtorig.shape[0], 3), np.float32)
    for joint in range(gtorig.shape[0]):
        t=transformPoint2D(gtorig[joint], M)
        gtcrop[joint, 0] = t[0]
        gtcrop[joint, 1] = t[1]
        gtcrop[joint, 2] = gtorig[joint, 2]
                
    if not doUseLabel:
        gtorig = np.zeros(gtorig.shape, gtorig.dtype)
        gtcrop = np.zeros(gtcrop.shape, gtcrop.dtype)
        gt3Dorig = np.zeros(gt3Dorig.shape, gt3Dorig.dtype)
        gt3Dcrop = np.zeros(gt3Dcrop.shape, gt3Dcrop.dtype)
        
    sampleConfig = copy.deepcopy(config)
    sampleConfig['cube'] = (cubesize[0], cubesize[1], cubesize[2])
    data.append(ICVLFrame(
        dpt.astype(np.float32),gtorig,gtcrop,M,gt3Dorig,
        gt3Dcrop,com3D,dptFileName,'',sampleConfig) )
        
    return NamedImgSequence(seqName,data,sampleConfig), jitter
    

def pointsImgTo3D(sample, fx, fy, ux, uy):
    """
    Normalize sample to metric 3D (NYU dataset specific, cf. code from Tompson)
    :param sample: points in (x,y,z) with x,y in image coordinates and z in mm
    z is assumed to be the distance from the camera plane (i.e., not camera center)
    :return: normalized points in mm
    """
    return pointsImgTo3D_NYU(sample, fx, fy, ux, uy)


def jointImgTo3D(sample, fx, fy, ux, uy):
    """
    Normalize sample to metric 3D (NYU dataset specific, cf. code from Tompson)
    :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
    :return: normalized joints in mm
    """
    return pointImgTo3D_NYU(sample, fx, fy, ux, uy)
    
    
def normalizeZeroOne(sample):
    imgD = np.asarray(sample.dpt.copy(), 'float32')
    imgD[imgD == 0] = sample.com[2] + (sample.config['cube'][2] / 2.)
    imgD -= (sample.com[2] - (sample.config['cube'][2] / 2.))
    imgD /= sample.config['cube'][2]
    
    target = np.clip(
                np.asarray(sample.gt3Dcrop, dtype='float32') 
                / sample.config['cube'][2], -0.5, 0.5) + 0.5
                
    return imgD, target
    
    
def normalizeMinusOneOne(sample):
    imgD = np.asarray(sample.dpt.copy(), 'float32')
    imgD[imgD == 0] = sample.com[2] + (sample.config['cube'][2] / 2.)
    imgD -= sample.com[2]
    imgD /= (sample.config['cube'][2] / 2.)
    
    target = np.clip(
                np.asarray(sample.gt3Dcrop, dtype='float32') 
                / (sample.config['cube'][2] / 2.), -1, 1)
                
    return imgD, target
    
        
def denormalizeJointPositions(jointPos, deNormZeroOne, cubesize):
    """
    Re-scale the given joint positions to metric distances (in mm)
    
    Arguments:
        jointPos (numpy.ndarray): normalized joint positions, 
            as provided by the dataset
        deNormZeroOne (boolean): whether jointPos are [0,1] normalized or [-1,1]
        cubesize: hand cube size
    """
    if (not len(jointPos.shape) == 3) \
            or (not len(cubesize.shape) == 2) \
            or (not jointPos.shape[0] == cubesize.shape[0]) \
            or (not jointPos.shape[2] == cubesize.shape[1]):
        raise UserWarning("Assumptions on input array dimensions not held")
        
    offset = 0
    scaleFactor = cubesize / 2.0
    if deNormZeroOne:
        offset = -0.5
        scaleFactor = cubesize
        
    scaleFactor = np.expand_dims(scaleFactor, axis=1)
        
    return ((jointPos + offset) * scaleFactor)
    
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

from collections import namedtuple
from enum import IntEnum

ICVLFrame = namedtuple('ICVLFrame',
                       ['dpt','gtorig','gtcrop','T','gt3Dorig',
                       'gt3Dcrop','com','fileName','subSeqName','config'])
NamedImgSequence = namedtuple('NamedImgSequence',['name','data','config'])
TrainLoaders = namedtuple('TrainLoaders', ['train_loader', 
                                           'loader_pretrain', 
                                           'loader_corr', 
                                           'loader_real', 
                                           'loader_synth', 
                                           'loader_real_weakcorr_ul', 
                                           'loader_synth_weakcorr_ul', 
                                           'loader_preview'])
                                          

class Camera(object):
    """
    Just encapsulating some camera information/parameters
    """
    
    def __init__(self, camid=0, fx=None, fy=None, ux=None, uy=None):
        self.camid = camid
        self.fx = fx
        self.fy = fy
        self.ux = ux
        self.uy = uy


class TrainingType(IntEnum):
    STANDARD = 0
    ADVERSARIAL = 2
    BATCHNORM_ONLY = 3
    FINETUNE_LAYER_DEPENDENT_LR = 4
    
    @classmethod
    def is_training_type_adversarial(self, training_type):
        return_val = False
        if training_type == self.ADVERSARIAL:
            return_val = True
        return return_val

        
class NetType(IntEnum):
    HAPE_DCGAN = 0
    HAPE_RESNET50 = 1
    HAPE_RESNET50_MARKUS = 2
    HAPE_RESNET50_FRANZISKA = 3
    HAPE_RESNET50_MARKUS_MAP = 4
    HAPE_RESNET50_MARKUS_MAP_PREVIEW = 5
    HAPE_RESNET50_MARKUS_A = 6
    HAPE_RESNET50_MAP_PREVIEW = 7
    
    @classmethod
    def get_num_output_views(self, net_type):
        return_val = 0
        if net_type == self.HAPE_RESNET50_MARKUS_MAP_PREVIEW \
            or net_type == self.HAPE_RESNET50_MAP_PREVIEW:
            return_val = 1
        return return_val
        
        
class DiscriminatorNetType(IntEnum):
    LINEAR = 0
    RESNET = 1
    
    
class LoaderMode(IntEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2
    
    
class DatasetType(IntEnum):
    NYU = 0
    ICG = 1


class ICVLPairStacks(object):
    '''
    Struct to hold two stacks of image data and a label indicating if the individual pairs are equal/similar or unequal/dissimilar
    '''
    def __init__(self,x0,x1,y0,y1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        
    def normalize(self,mean,std):
        self.x0 -= mean
        self.x0 /= std
        self.x1 -= mean
        self.x1 /= std


class ImgStackAndIndices(object):
    '''
    Struct to hold a stacks of image data and labels indicating target class, pairs of similar and dissimilar samples and triplets of sim/dissim
    '''
    def __init__(self,x,y=None,sampleInfo=None,pairIdx=None,pairLabels=None,tripletIdx=None,nPairsPerBatch=None,nTripletsPerBatch=None,batchSize=None,y2=None):
        self.x = x
        self.y = y
        self.y2 = y2
        self.sampleInfo = sampleInfo  #  additional stuff like pose, ...
        self.pairIdx = pairIdx
        self.pairLabels = pairLabels
        self.tripletIdx = tripletIdx
        self.nPairsPerBatch = nPairsPerBatch
        self.nTripletsPerBatch = nTripletsPerBatch
        self.batchSize = batchSize
        
        
class Jitter(object):
    """
    Collection of jitter parameters (i.e., random numbers for specific jitter 
    "actions"). E.g., to re-produce a specific jittered example.
    """
    def __init__(self):
        self.img_white_noise_scale = None
        self.rotation_angle_scale = None
        self.detection_offset_scale = None
        self.crop_scale = None
        
        
class Arguments(object):
    pass




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

import torch

import argparse


def parse_arguments_generic(args=None):
    """
    Parses command-line arguments of the generic train/test script
    
    Arguments:
        args (Object, optional): existing object with attributes to which the 
            parsed arguments are added (default: None)
    """
    parser = argparse.ArgumentParser(description='Model training/evaluation')
    parser = add_definitions_for_generic_script(parser)
    args = parser.parse_args(namespace=args)
    
    # Adapt necessary parameters (names, types, ...)
    args.do_train = not args.no_train
    args.do_test = not args.no_test
    args.crop_size_3d_tuple = (args.crop_size_3d, args.crop_size_3d, args.crop_size_3d)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    return args
    

def add_definitions_for_generic_script(parser):
    parser.add_argument('--dataset-type', type=int, default=0,
                        help='Dataset Type (see data.basetypes.DatasetType)')
    parser.add_argument('--net-type', type=int, default=7,
                        help='Network Type (see data.basetypes.NetType)')
    parser.add_argument('--do-load-pretrained-model', action='store_true', default=False,
                        help='enables loading of a pre-trained model \
                        which is used to initialize the model')
    parser.add_argument('--optim-type', type=int, default=0,
                        help='Optimizer (e.g., 0: Adam (default), 1: RMSprop, 2: SGD)')
    parser.add_argument('--lr', type=float, default=3.3*1e-4, metavar='g',
                        help='learning rate (default: 3.3*1e-4)')
    parser.add_argument('--lambda-embedding-loss', type=float, default=0.2, metavar='l_e',
                        help='weight for embedding loss term, which enforces \
                        the embedding of corresponding real and synthetic samples to be equal (default: 0.2)')
    parser.add_argument('--lambda-realdata-loss', type=float, default=1.0, metavar='l_e',
                        help='weight for (joint positions) loss on real data (default: 1.0)')
    parser.add_argument('--training-type', type=int, default=2,
                        help='Training Type. (see data.basetypes.TrainingType)')
    parser.add_argument('--lambda-adversarial-loss', type=float, default=1e-5, metavar='l_a',
                        help='weight for adversarial loss term on embedding (default: 1e-5)')
    parser.add_argument('--lambda-preview-loss', type=float, default=1e-4, metavar='l_p',
                        help='weight for view prediction loss term (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.001, metavar='d',
                        help='weight decay (default: 1e-3)')
    parser.add_argument('--no-backprop-through-featext-for-emb', action='store_true', default=True,
                        help='do not back-propagate through feature extractor \
                        for embedding loss (default: set true)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=120, metavar='n_e',
                        help='number of epochs to train (default: 120)')
    parser.add_argument('--num-labeled-samples', type=int, default=10000000, metavar='N',
                        help='(max.) number of labeled training samples to use \
                        (default: 10000000)')
    parser.add_argument('--ratio-corresponding-data-in-batch', type=float, default=0.33, metavar='r_c',
                        help='default: 0.33')
    parser.add_argument('--ratio-synth-data-in-batch', type=float, default=0.33, metavar='r_s',
                        help='default: 0.33')
    parser.add_argument('--ratio-unlabeled-data-in-batch', type=float, default=0.33, metavar='r_u',
                        help='default: 0.33')
    parser.add_argument('--crop-size-3d', type=int, default=300,
                        help='crop size in 3d, in mm, same is used in each dimension \
                        (default: 300).')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpu-id', type=int, default=0, metavar='D',
                        help='ID of the GPU to be used (for "multi-GPU" machine, default: 0)')
    parser.add_argument('--seed', type=int, default=123456789, metavar='S',
                        help='random seed (default: 123456789)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='n_i',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--exp-name', default="run00_default",
                        help='name for the experiment (default: "run00_default")')
    parser.add_argument('--no-train', action='store_true', default=False,
                        help='do not train (default: not set, i.e., do training)')
    parser.add_argument('--no-test', action='store_true', default=False,
                        help='do not test (default: not set, i.e., do testing)')
    parser.add_argument('--model-filepath', default="",
                        help='filename (and full path) to store/load the model.\
                        default location is within the respective result folder \
                        (in folder model: "./model/model.mdl").')
    parser.add_argument('--out-base-path', default="",
                        help='(absolute) base path for --out-path; default = "" \
                        meaning that the directory containing the script is used.')
    parser.add_argument('--out-path', default="../results/default",
                        help='relative path to store outputs (results, model, ...). \
                        (default: "<out-base-path>/../results/default"')
                        
    return parser
    
    